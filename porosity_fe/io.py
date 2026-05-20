"""JSON / VTK I/O helpers and provenance."""

import dataclasses
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from .results import ConfigArtifacts

from .results import ConfigResult

logger = logging.getLogger("porosity_fe_analysis")

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
        # Source checkout not pip-installed: fall back to the package
        # ``__version__`` attribute (defined in ``porosity_fe/__init__.py``).
        from . import __version__ as _pkg_version_attr
        pfe_version = _pkg_version_attr

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
