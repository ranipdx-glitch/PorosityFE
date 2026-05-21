#!/usr/bin/env python3
"""End-to-end JSON output contract tests (#20).

Exercises the full save -> jsonschema.validate -> load_results_from_json
loop on a minimal one-Vp / one-config sweep so any drift in the envelope,
schema, or loader is caught in a single test file.
"""

import dataclasses
import json
import os

import numpy as np
import pytest

from porosity_fe_analysis import (
    FORMAT_EMPIRICAL_SWEEP,
    FORMAT_NCR,
    JSON_SCHEMA_VERSION,
    POROSITY_CONFIGS,
    _json_default,
    compare_configurations,
    load_results_from_json,
    save_results_to_json,
)


_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "validation",
    "schemas",
    "porosity_results_schema.json",
)


def _load_schema():
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _tiny_results():
    """Smallest sweep that still produces a real payload."""
    return compare_configurations(
        0.03,
        configs={"uniform_spherical": POROSITY_CONFIGS["uniform_spherical"]},
    )


def test_save_validate_load_round_trip(tmp_path):
    import jsonschema

    path = str(tmp_path / "round_trip.json")
    save_results_to_json(_tiny_results(), path)

    with open(path, encoding="utf-8") as f:
        on_disk = json.load(f)

    # Schema validation: catches envelope drift.
    jsonschema.validate(instance=on_disk, schema=_load_schema())

    # Loader returns the same dict (apart from object identity).
    loaded = load_results_from_json(path)
    assert loaded == on_disk

    # Envelope sanity.
    assert loaded["schema_version"] == JSON_SCHEMA_VERSION
    assert loaded["format"] == FORMAT_EMPIRICAL_SWEEP
    assert "provenance" in loaded
    assert "uniform_spherical" in loaded


def test_ncr_format_validates_against_schema(tmp_path):
    """The NCR exporter is one of the three known top-level formats and
    must validate against the same shared schema (#20)."""
    import jsonschema

    from porosity_fe.reporting import build_ncr_record, write_ncr_json

    result = {
        "config": {
            "material_name": "T800_epoxy",
            "n_plies": 24,
            "t_ply": 0.183,
            "Vp": 3.0,
            "distribution": "uniform",
            "void_shape": "spherical",
            "nx": 30, "ny": 10, "nz": 12,
        },
        "empirical": {
            "compression": {
                "judd_wright": {"failure_stress": 1234.5, "knockdown": 0.823},
            },
            "ilss": {
                "judd_wright": {"failure_stress": 67.0, "knockdown": 0.744},
            },
        },
    }
    meta = {
        "prepared_by": "test",
        "ncr_reference": "NCR-2026-0001",
        "structural_class": "primary",
        "date": "2026-05-19",
        "layup": "[0/90]_s",
    }
    path = str(tmp_path / "ncr.json")
    write_ncr_json(path, build_ncr_record(result, meta))

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    jsonschema.validate(instance=doc, schema=_load_schema())
    assert doc["format"] == FORMAT_NCR

    loaded = load_results_from_json(path)
    assert loaded["format"] == FORMAT_NCR


def test_numpy_default_handler_round_trips_ndarray(tmp_path):
    """An ndarray smuggled into a config dict must serialise via
    _json_default rather than raise TypeError (#20 item 4)."""
    results = _tiny_results()
    # #44: compare_configurations now returns ConfigResult dataclasses, so
    # mutate via dataclasses.replace rather than dict() (the dict-protocol
    # shim drops the live-object keys and would break save_results_to_json).
    original = results["uniform_spherical"]
    replacement = dataclasses.replace(
        original,
        config={
            **original.config,
            "ply_angles_deg": np.array([0.0, 45.0, -45.0, 90.0]),
            "n_plies_np": np.int64(12),
        },
    )
    results = {"uniform_spherical": replacement}

    path = str(tmp_path / "with_ndarray.json")
    save_results_to_json(results, path)

    loaded = load_results_from_json(path)
    assert loaded["uniform_spherical"]["config"]["ply_angles_deg"] == [
        0.0,
        45.0,
        -45.0,
        90.0,
    ]
    assert loaded["uniform_spherical"]["config"]["n_plies_np"] == 12


def test_json_default_handles_dataclass():
    """Plain dataclass instances must be serialisable by _json_default
    (#20 item 4: numpy-type fragility)."""

    @dataclasses.dataclass
    class _Foo:
        a: int
        b: str

    out = _json_default(_Foo(a=1, b="x"))
    assert out == {"a": 1, "b": "x"}


def test_json_default_rejects_unknown_type():
    class _Opaque:
        pass

    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_default(_Opaque())


def test_save_results_hostname_opt_in_via_env(tmp_path, monkeypatch):
    """#152: ``provenance.hostname`` is opt-in via
    ``POROSITY_FE_INCLUDE_HOSTNAME``. When the env var is set the field
    is populated with a non-empty string; when unset the key is omitted
    from the provenance block entirely (no workstation-name leakage).
    """
    results = _tiny_results()

    # Opt-in: env var set to a truthy value -> hostname populated.
    monkeypatch.setenv("POROSITY_FE_INCLUDE_HOSTNAME", "1")
    on_path = str(tmp_path / "with_host.json")
    save_results_to_json(results, on_path)
    with open(on_path, encoding="utf-8") as f:
        prov_on = json.load(f)["provenance"]
    assert "hostname" in prov_on
    assert isinstance(prov_on["hostname"], str)
    assert prov_on["hostname"]  # non-empty

    # Default: env var unset -> hostname key absent from the provenance.
    monkeypatch.delenv("POROSITY_FE_INCLUDE_HOSTNAME", raising=False)
    off_path = str(tmp_path / "no_host.json")
    save_results_to_json(results, off_path)
    with open(off_path, encoding="utf-8") as f:
        prov_off = json.load(f)["provenance"]
    assert "hostname" not in prov_off


def test_save_results_config_name_collision_raises(tmp_path):
    """#152: a user-supplied ConfigResult keyed with one of the reserved
    envelope keys (``'schema_version'`` / ``'format'``) must raise
    ValueError rather than silently overwrite the envelope on disk.
    """
    results = _tiny_results()
    original = next(iter(results.values()))
    colliding = {"schema_version": original}

    path = str(tmp_path / "collision.json")
    with pytest.raises(ValueError, match="collides"):
        save_results_to_json(colliding, path)


def test_save_results_legacy_dict_shape_still_works(tmp_path):
    """#152: the pre-#103 ``Dict[str, dict]`` shape (raw worker-dict with
    ``porosity_field`` / ``config`` / ``empirical`` keys) must still
    round-trip through save_results_to_json so callers that haven't
    migrated to ConfigResult keep working.
    """
    # Borrow real numbers from a tiny sweep, then re-shape them into the
    # legacy worker-dict that predates ConfigResult.
    cr = _tiny_results()["uniform_spherical"]

    class _PFStub:
        # Legacy path reads only ``.Vp`` off the porosity_field object.
        Vp = cr.Vp
        seed = cr.seed

    legacy_results = {
        "uniform_spherical": {
            "porosity_field": _PFStub(),
            "config": cr.config,
            "empirical": cr.empirical,
        },
    }

    path = str(tmp_path / "legacy.json")
    save_results_to_json(legacy_results, path)

    loaded = load_results_from_json(path)
    assert loaded["schema_version"] == JSON_SCHEMA_VERSION
    assert loaded["format"] == FORMAT_EMPIRICAL_SWEEP
    assert "uniform_spherical" in loaded
    assert loaded["uniform_spherical"]["void_volume_fraction"] == float(cr.Vp)
    # Empirical knockdown table survives the round-trip.
    assert (
        loaded["uniform_spherical"]["empirical"]["compression"]["judd_wright"]
        ["knockdown"]
        == cr.empirical["compression"]["judd_wright"]["knockdown"]
    )
