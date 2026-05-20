"""Backwards-compatibility shim.

All implementation moved to the :mod:`porosity_fe` package as part of
issue #119 (splitting the 6800-line god-file into focused submodules).
This module is preserved so existing imports of the form
``from porosity_fe_analysis import MaterialProperties, ...`` (used by
``app.py``, ``validate_porosity_cli.py``, ``validation/validate_all.py``,
the example scripts, the test suite and the Sphinx docs) keep working
unchanged.

New code should import directly from :mod:`porosity_fe`::

    from porosity_fe import MaterialProperties, EmpiricalSolver, FESolver

The shim is intentionally exhaustive: every public symbol that any call
site historically imported from ``porosity_fe_analysis`` is re-exported
here (including the underscored names a handful of tests reach into,
such as ``_json_default``, ``_build_provenance`` and ``_resolve_ply_angles``).
"""

from porosity_fe import *  # noqa: F401, F403
from porosity_fe import _FATIGUE_B_QI as _FATIGUE_B_QI  # noqa: F401
from porosity_fe import _FATIGUE_KD_FLOOR as _FATIGUE_KD_FLOOR  # noqa: F401
from porosity_fe import _KNOWN_FORMATS as _KNOWN_FORMATS  # noqa: F401
from porosity_fe import _MT_CACHE_MAXSIZE as _MT_CACHE_MAXSIZE  # noqa: F401
from porosity_fe import _NODE_COORDS_REF as _NODE_COORDS_REF  # noqa: F401
from porosity_fe import _PLY_ANGLES_QI as _PLY_ANGLES_QI  # noqa: F401
from porosity_fe import _PLY_ANGLES_UD as _PLY_ANGLES_UD  # noqa: F401
from porosity_fe import _UQ_DEFAULT_PERCENTILES as _UQ_DEFAULT_PERCENTILES  # noqa: F401
from porosity_fe import DEFAULT_POROSITY_LEVELS as DEFAULT_POROSITY_LEVELS  # noqa: F401

# Re-export the underscored / non-``__all__`` names that the test suite,
# the Streamlit app, and the example scripts reach into.  ``from X import *``
# only pulls names listed in ``__all__``, so private helpers and the
# uncategorised constants below need an explicit re-import.
from porosity_fe import __version__ as __version__  # noqa: F401
from porosity_fe import _analyze_one as _analyze_one  # noqa: F401
from porosity_fe import _apply_plot_style as _apply_plot_style  # noqa: F401
from porosity_fe import _build_arg_parser as _build_arg_parser  # noqa: F401
from porosity_fe import _build_clt_abd as _build_clt_abd  # noqa: F401
from porosity_fe import _build_config_artifacts as _build_config_artifacts  # noqa: F401
from porosity_fe import _build_config_result as _build_config_result  # noqa: F401
from porosity_fe import _build_provenance as _build_provenance  # noqa: F401
from porosity_fe import _configure_cli_logging as _configure_cli_logging  # noqa: F401
from porosity_fe import _configure_matplotlib_style as _configure_matplotlib_style  # noqa: F401
from porosity_fe import _degraded_composite_stiffness as _degraded_composite_stiffness  # noqa: F401
from porosity_fe import _draw_unit_samples as _draw_unit_samples  # noqa: F401
from porosity_fe import _DynamicStdoutHandler as _DynamicStdoutHandler  # noqa: F401
from porosity_fe import _json_default as _json_default  # noqa: F401
from porosity_fe import _mt_cache as _mt_cache  # noqa: F401
from porosity_fe import _mt_cache_clear as _mt_cache_clear  # noqa: F401
from porosity_fe import _mt_cache_key as _mt_cache_key  # noqa: F401
from porosity_fe import _mt_effective_stiffness as _mt_effective_stiffness  # noqa: F401
from porosity_fe import _normalize_uq_spec as _normalize_uq_spec  # noqa: F401
from porosity_fe import _resolve_n_jobs as _resolve_n_jobs  # noqa: F401
from porosity_fe import _resolve_ply_angles as _resolve_ply_angles  # noqa: F401
from porosity_fe import _vp_label as _vp_label  # noqa: F401
from porosity_fe import main as _main  # noqa: F401

if __name__ == "__main__":  # pragma: no cover - parity with the old entry point
    import sys
    sys.exit(_main())
