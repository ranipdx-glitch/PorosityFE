"""``porosity_fe`` — porosity-degraded composite-laminate analysis (#119).

This package is the new home of every public symbol that used to live in
the monolithic ``porosity_fe_analysis.py``. The split keeps the public
API exactly stable: a thin compatibility shim at
``porosity_fe_analysis.py`` re-exports everything from here, so callers
that import ``from porosity_fe_analysis import X`` continue to work.

New code should import directly from this package::

    from porosity_fe import MaterialProperties, EmpiricalSolver

The matplotlib style helper from #53 is applied once at package import
time so any module that imports :mod:`porosity_fe` inherits the same
rcParams.
"""

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Version: identical to what the old monolith used to expose. Kept here
# so ``porosity_fe.__version__`` (and the shim's re-export) keep working.
# ----------------------------------------------------------------------
try:  # pragma: no cover - exercised by both source and pip-installed paths
    import importlib.metadata as _ilm
    __version__ = _ilm.version("porosity-fe")
except Exception:  # pragma: no cover - exercised in source checkouts
    # Source checkout that isn't pip-installed. Keep in sync with
    # pyproject.toml on each release.
    __version__ = "1.2.0"

# ----------------------------------------------------------------------
# Apply shared rcParams + cache the LABEL_* / LABELS constants. Run at
# package-import time so any importer of ``porosity_fe`` (the Streamlit
# app, validation runner, tests) inherits the same plot style.
# ----------------------------------------------------------------------
from ._style import LABEL_KNOCKDOWN as LABEL_KNOCKDOWN  # noqa: F401, E402
from ._style import LABEL_MAE_PCT as LABEL_MAE_PCT  # noqa: F401, E402
from ._style import LABEL_POROSITY_PCT as LABEL_POROSITY_PCT  # noqa: F401, E402
from ._style import LABEL_POROSITY_VP as LABEL_POROSITY_VP  # noqa: F401, E402
from ._style import LABEL_SCF as LABEL_SCF  # noqa: F401, E402
from ._style import LABEL_STIFFNESS_RETENTION as LABEL_STIFFNESS_RETENTION  # noqa: F401, E402
from ._style import LABEL_STIFFNESS_RETENTION_FRAC as LABEL_STIFFNESS_RETENTION_FRAC  # noqa: F401, E402
from ._style import LABEL_STRESS_MPA as LABEL_STRESS_MPA  # noqa: F401, E402
from ._style import LABEL_X_MM as LABEL_X_MM  # noqa: F401, E402
from ._style import LABEL_Y_MM as LABEL_Y_MM  # noqa: F401, E402
from ._style import LABEL_Z_MM as LABEL_Z_MM  # noqa: F401, E402
from ._style import LABELS as LABELS  # noqa: F401, E402
from ._style import _apply_plot_style as _apply_plot_style  # noqa: F401, E402
from ._style import _configure_matplotlib_style as _configure_matplotlib_style  # noqa: F401, E402

_configure_matplotlib_style()

# ----------------------------------------------------------------------
# Re-export the full public surface. Order roughly follows the layered
# package layout (low-level dataclasses first, sweep orchestrator last).
# Each line uses ``as X`` self-aliases + ``noqa: F401, E402`` so ruff
# treats these as explicit re-exports rather than unused imports.
# ----------------------------------------------------------------------
from ._ply_angles import _PLY_ANGLES_QI as _PLY_ANGLES_QI  # noqa: F401, E402
from ._ply_angles import _PLY_ANGLES_UD as _PLY_ANGLES_UD  # noqa: F401, E402
from ._ply_angles import _resolve_ply_angles as _resolve_ply_angles  # noqa: F401, E402
from .cli import DEFAULT_POROSITY_LEVELS as DEFAULT_POROSITY_LEVELS  # noqa: F401, E402
from .cli import _build_arg_parser as _build_arg_parser  # noqa: F401, E402
from .cli import _configure_cli_logging as _configure_cli_logging  # noqa: F401, E402
from .cli import _DynamicStdoutHandler as _DynamicStdoutHandler  # noqa: F401, E402
from .cli import _vp_label as _vp_label  # noqa: F401, E402
from .cli import main as main  # noqa: F401, E402
from .empirical import EmpiricalSolver as EmpiricalSolver  # noqa: F401, E402
from .fatigue import _FATIGUE_B_QI as _FATIGUE_B_QI  # noqa: F401, E402
from .fatigue import _FATIGUE_KD_FLOOR as _FATIGUE_KD_FLOOR  # noqa: F401, E402
from .fatigue import FatigueModel as FatigueModel  # noqa: F401, E402
from .fe import BoundaryHandler as BoundaryHandler  # noqa: F401, E402
from .fe import FESolver as FESolver  # noqa: F401, E402
from .fe import FieldResults as FieldResults  # noqa: F401, E402
from .fe import GlobalAssembler as GlobalAssembler  # noqa: F401, E402
from .fe import Hex8Element as Hex8Element  # noqa: F401, E402
from .fe.element import _NODE_COORDS_REF as _NODE_COORDS_REF  # noqa: F401, E402
from .gauss import gauss_points_1d as gauss_points_1d  # noqa: F401, E402
from .gauss import gauss_points_hex as gauss_points_hex  # noqa: F401, E402
from .homogenization import _MT_CACHE_MAXSIZE as _MT_CACHE_MAXSIZE  # noqa: F401, E402
from .homogenization import _build_clt_abd as _build_clt_abd  # noqa: F401, E402
from .homogenization import _degraded_composite_stiffness as _degraded_composite_stiffness  # noqa: F401, E402
from .homogenization import _mt_cache as _mt_cache  # noqa: F401, E402
from .homogenization import _mt_cache_clear as _mt_cache_clear  # noqa: F401, E402
from .homogenization import _mt_cache_key as _mt_cache_key  # noqa: F401, E402
from .homogenization import _mt_effective_stiffness as _mt_effective_stiffness  # noqa: F401, E402
from .homogenization import compute_clt_effective_modulus as compute_clt_effective_modulus  # noqa: F401, E402
from .homogenization import (  # noqa: F401, E402
    compute_degraded_clt_flexural_modulus as compute_degraded_clt_flexural_modulus,
)
from .homogenization import compute_degraded_clt_moduli as compute_degraded_clt_moduli  # noqa: F401, E402
from .io import _KNOWN_FORMATS as _KNOWN_FORMATS  # noqa: F401, E402
from .io import FORMAT_EMPIRICAL_SWEEP as FORMAT_EMPIRICAL_SWEEP  # noqa: F401, E402
from .io import FORMAT_FE_FIELDS as FORMAT_FE_FIELDS  # noqa: F401, E402
from .io import FORMAT_NCR as FORMAT_NCR  # noqa: F401, E402
from .io import JSON_SCHEMA_VERSION as JSON_SCHEMA_VERSION  # noqa: F401, E402
from .io import _build_provenance as _build_provenance  # noqa: F401, E402
from .io import _json_default as _json_default  # noqa: F401, E402
from .io import load_results_from_json as load_results_from_json  # noqa: F401, E402
from .io import save_results_to_json as save_results_to_json  # noqa: F401, E402
from .materials import MATERIALS as MATERIALS  # noqa: F401, E402
from .materials import MaterialProperties as MaterialProperties  # noqa: F401, E402
from .mesh import CompositeMesh as CompositeMesh  # noqa: F401, E402
from .mesh import check_mesh_quality as check_mesh_quality  # noqa: F401, E402
from .pipeline import _analyze_one as _analyze_one  # noqa: F401, E402
from .pipeline import _build_config_artifacts as _build_config_artifacts  # noqa: F401, E402
from .pipeline import _build_config_result as _build_config_result  # noqa: F401, E402
from .pipeline import _resolve_n_jobs as _resolve_n_jobs  # noqa: F401, E402
from .pipeline import compare_configurations as compare_configurations  # noqa: F401, E402
from .porosity_field import POROSITY_CONFIGS as POROSITY_CONFIGS  # noqa: F401, E402
from .porosity_field import PorosityField as PorosityField  # noqa: F401, E402
from .results import ConfigArtifacts as ConfigArtifacts  # noqa: F401, E402
from .results import ConfigResult as ConfigResult  # noqa: F401, E402
from .results import FailureResult as FailureResult  # noqa: F401, E402
from .transforms import rotate_stiffness_3d as rotate_stiffness_3d  # noqa: F401, E402
from .transforms import rotation_matrix_3d as rotation_matrix_3d  # noqa: F401, E402
from .transforms import strain_transformation_3d as strain_transformation_3d  # noqa: F401, E402
from .transforms import stress_transformation_3d as stress_transformation_3d  # noqa: F401, E402
from .uq import _UQ_DEFAULT_PERCENTILES as _UQ_DEFAULT_PERCENTILES  # noqa: F401, E402
from .uq import _draw_unit_samples as _draw_unit_samples  # noqa: F401, E402
from .uq import _normalize_uq_spec as _normalize_uq_spec  # noqa: F401, E402
from .uq import propagate_uncertainty as propagate_uncertainty  # noqa: F401, E402
from .viz import FEVisualizer as FEVisualizer  # noqa: F401, E402
from .void_geometry import VOID_SHAPES as VOID_SHAPES  # noqa: F401, E402
from .void_geometry import VoidGeometry as VoidGeometry  # noqa: F401, E402

__all__ = [
    # Plot style / label constants (#53)
    "LABEL_KNOCKDOWN",
    "LABEL_MAE_PCT",
    "LABEL_POROSITY_PCT",
    "LABEL_POROSITY_VP",
    "LABEL_SCF",
    "LABEL_STIFFNESS_RETENTION",
    "LABEL_STIFFNESS_RETENTION_FRAC",
    "LABEL_STRESS_MPA",
    "LABEL_X_MM",
    "LABEL_Y_MM",
    "LABEL_Z_MM",
    "LABELS",
    # Materials
    "MATERIALS",
    "MaterialProperties",
    # Void geometry
    "VOID_SHAPES",
    "VoidGeometry",
    # Porosity field
    "POROSITY_CONFIGS",
    "PorosityField",
    # Mesh
    "CompositeMesh",
    "check_mesh_quality",
    # Result dataclasses
    "ConfigArtifacts",
    "ConfigResult",
    "FailureResult",
    # Empirical / fatigue / UQ
    "EmpiricalSolver",
    "FatigueModel",
    "propagate_uncertainty",
    # Coordinate transforms
    "rotation_matrix_3d",
    "rotate_stiffness_3d",
    "strain_transformation_3d",
    "stress_transformation_3d",
    # Gauss
    "gauss_points_1d",
    "gauss_points_hex",
    # Homogenization / CLT
    "compute_clt_effective_modulus",
    "compute_degraded_clt_moduli",
    "compute_degraded_clt_flexural_modulus",
    # FE
    "BoundaryHandler",
    "FESolver",
    "FieldResults",
    "GlobalAssembler",
    "Hex8Element",
    # Visualization
    "FEVisualizer",
    # IO
    "FORMAT_EMPIRICAL_SWEEP",
    "FORMAT_FE_FIELDS",
    "FORMAT_NCR",
    "JSON_SCHEMA_VERSION",
    "load_results_from_json",
    "save_results_to_json",
    # Pipeline / CLI
    "compare_configurations",
    "main",
    # Version
    "__version__",
]
