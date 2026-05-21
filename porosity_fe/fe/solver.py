"""FE solver and FieldResults dataclass."""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .._ply_angles import _resolve_ply_angles
from ..homogenization import _mt_effective_stiffness
from ..io import (
    FORMAT_FE_FIELDS,
    JSON_SCHEMA_VERSION,
    _build_provenance,
    _json_default,
)
from ..materials import MaterialProperties
from ..mesh import CompositeMesh, check_mesh_quality
from ..porosity_field import PorosityField
from ..results import FailureResult
from ..transforms import rotate_stiffness_3d, strain_transformation_3d, stress_transformation_3d
from .assembler import BoundaryHandler, GlobalAssembler

logger = logging.getLogger("porosity_fe_analysis")

# ============================================================
# SECTION 7g: FE SOLVER AND FIELD RESULTS
# ============================================================

@dataclass
class FieldResults:
    """Results from a finite element solve.

    Attributes
    ----------
    displacement : np.ndarray
        Shape (n_nodes, 3) nodal displacements.
    stress_global : np.ndarray
        Shape (n_elem, n_gp, 6) stress in global coordinates.
    stress_local : np.ndarray
        Shape (n_elem, n_gp, 6) stress in local (material) coordinates.
    strain_global : np.ndarray
        Shape (n_elem, n_gp, 6) strain in global coordinates.
    strain_local : np.ndarray
        Shape (n_elem, n_gp, 6) strain in local coordinates.
    max_failure_index : float
        Maximum failure index across all Gauss points (criterion-dependent).
    knockdown : float
        Stiffness knockdown factor (modulus ratio: E_porous/E_pristine).
    per_element_failure_index : np.ndarray or None
        Shape (n_elem,) max-over-Gauss-point failure index per element.
        Optional (defaults to ``None`` for back-compatibility with callers
        that construct ``FieldResults`` directly); populated by
        ``FESolver.solve`` and consumed by the VTK export so failure
        hot-spots can be sliced in ParaView.
    failure_criterion : str
        Which failure criterion was used to produce ``max_failure_index`` and
        ``per_element_failure_index``. One of ``'tsai_wu'`` (default for
        back-compat), ``'hashin'``, or ``'max_stress'``.
    failure_mode_indices : dict or None
        Per-mode breakdown of the maximum failure index across the model with
        keys ``'fiber_t'``, ``'fiber_c'``, ``'matrix_t'``, ``'matrix_c'``,
        ``'shear'`` (plus ``'max_fi'``). For Tsai-Wu the per-mode entries are
        ``NaN`` (the polynomial does not separate modes); for ``max_stress``
        the unused entries are zero. Lets the GUI and JSON exporter report
        the dominant failure mode, not just severity.

    Notes
    -----
    Voigt order for both stress and strain: ``[11, 22, 33, 23, 13, 12]``
    (normals first, then 23 / 13 / 12 shears). The last three strain
    components are **engineering** strain (``gamma_ij = 2 * eps_ij``); the
    last three stress components are the matching shear stresses
    ``[tau_23, tau_13, tau_12]``. Sign convention: tensile normals are
    positive, compressive normals are negative — these arrays are signed
    (unlike the empirical ``failure_stress_MPa`` returned by
    :meth:`EmpiricalSolver.apply_loading`, which is a positive magnitude).
    Use :func:`strain_transformation_3d` / :func:`stress_transformation_3d`
    to rotate these arrays between frames.
    """
    displacement: np.ndarray
    stress_global: np.ndarray
    stress_local: np.ndarray
    strain_global: np.ndarray
    strain_local: np.ndarray
    max_failure_index: float
    knockdown: float
    per_element_failure_index: Optional[np.ndarray] = None
    failure_criterion: str = 'tsai_wu'
    failure_mode_indices: Optional[Dict[str, float]] = None

    def __repr__(self) -> str:
        n_nodes = self.displacement.shape[0] if self.displacement is not None else 0
        n_elem = self.stress_global.shape[0] if self.stress_global is not None else 0
        return (f"FieldResults(n_nodes={n_nodes}, n_elements={n_elem}, "
                f"max_FI={self.max_failure_index:.4f}, "
                f"knockdown={self.knockdown:.4f})")

    def summary(self, sigma_pristine: Optional[float] = None,
                model_label: Optional[str] = None) -> 'FailureResult':
        """Distill the field result into a :class:`FailureResult`.

        Unifies the FE return shape with the empirical solver (#44 item 1)
        so callers can treat the two solver outputs polymorphically.

        Parameters
        ----------
        sigma_pristine : float, optional
            Pristine reference stress (MPa) used to compute
            ``failure_stress = knockdown * sigma_pristine``. If ``None``
            (default), the ``failure_stress`` field is set to
            ``knockdown`` itself (unit knockdown — the bare ratio); pass
            the loading-mode-specific pristine strength
            (``material.sigma_1c`` for compression, ``material.tau_ilss``
            for ILSS, etc.) to get a meaningful magnitude.
        model_label : str, optional
            Label used for the ``model`` field. Defaults to
            ``f"fe_{self.failure_criterion}"`` so the FE summary is
            self-describing and distinguishable from the empirical labels.

        Returns
        -------
        FailureResult
            Unified summary with the FE ``knockdown``, derived
            ``failure_stress``, FE-criterion-tagged ``model`` and a
            ``details`` dict carrying ``max_failure_index``,
            ``failure_criterion`` and ``failure_mode_indices`` for
            downstream consumers that need the richer field-result data.
        """
        kd = float(self.knockdown)
        sigma_ref = float(sigma_pristine) if sigma_pristine is not None else 1.0
        return FailureResult(
            failure_stress=kd * sigma_ref,
            knockdown=kd,
            model=str(model_label) if model_label is not None
            else f"fe_{self.failure_criterion}",
            details={
                'max_failure_index': float(self.max_failure_index),
                'failure_criterion': self.failure_criterion,
                'failure_mode_indices': (
                    dict(self.failure_mode_indices)
                    if self.failure_mode_indices is not None else None
                ),
            },
        )

    def to_vtk(self, mesh: 'CompositeMesh', filename: str) -> None:
        """Write the hex mesh and per-element FE fields to a legacy ASCII VTK
        file (``UNSTRUCTURED_GRID``) for inspection in ParaView / VisIt / PyVista.

        The writer is dependency-free: it emits the legacy VTK 3.0 ASCII
        format by hand. The 8-node hex connectivity already stored in
        ``mesh.elements`` follows the standard VTK hexahedron ordering
        (bottom face CCW then top face CCW, see ``_NODE_COORDS_REF``), so the
        cells are written verbatim with cell type 12 (``VTK_HEXAHEDRON``).

        Point data
        -----------
        - ``displacement`` (3-vector), and the scalars ``porosity``,
          ``stiffness_reduction``, ``ply_id`` if present on the mesh.

        Cell data
        ---------
        - element-averaged ``von_mises`` and the six global stress
          (``sigma_xx`` .. ``tau_xy``) and strain (``eps_xx`` .. ``gamma_xy``)
          components (Gauss points reduced by mean),
        - ``tsai_wu_index`` (max-over-GP per element, if available),
        - ``Vp_elem`` (mean nodal porosity over the 8 corners),
        - ``ply_id``, ``ply_angle_deg``, ``is_void`` and ``knockdown`` where
          available.

        Parameters
        ----------
        mesh : CompositeMesh
            The mesh that produced these results (supplies geometry,
            connectivity, porosity and ply metadata).
        filename : str
            Output ``.vtk`` file path.
        """
        nodes = np.asarray(mesh.nodes, dtype=float)
        elements = np.asarray(mesh.elements, dtype=np.int64)
        n_nodes = nodes.shape[0]
        n_elem = elements.shape[0]

        if elements.shape[1] != 8:
            raise ValueError(
                f"to_vtk only supports 8-node hexahedra; got connectivity "
                f"of width {elements.shape[1]}."
            )
        if self.displacement is not None and \
                self.displacement.shape[0] != n_nodes:
            raise ValueError(
                f"displacement has {self.displacement.shape[0]} rows but the "
                f"mesh has {n_nodes} nodes; results do not match this mesh."
            )

        # Gauss-point-averaged global stress/strain -> (n_elem, 6)
        sig = np.mean(self.stress_global, axis=1)
        eps = np.mean(self.strain_global, axis=1)

        # Element-averaged von Mises from the averaged stress tensor.
        sxx, syy, szz = sig[:, 0], sig[:, 1], sig[:, 2]
        tyz, txz, txy = sig[:, 3], sig[:, 4], sig[:, 5]
        von_mises = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2)
        )

        def _fmt(values) -> str:
            return "\n".join(repr(float(v)) for v in np.asarray(values).ravel())

        def _fmt_xyz(rows) -> str:
            return "\n".join(
                f"{float(r[0])!r} {float(r[1])!r} {float(r[2])!r}"
                for r in rows
            )

        lines = [
            "# vtk DataFile Version 3.0",
            "PorosityFE results (hex mesh + per-element fields)",
            "ASCII",
            "DATASET UNSTRUCTURED_GRID",
            f"POINTS {n_nodes} float",
        ]
        lines.append(_fmt_xyz(nodes))

        # CELLS: each line is "8 n0 n1 ... n7"; total size = n_elem * 9.
        lines.append(f"CELLS {n_elem} {n_elem * 9}")
        lines.append(
            "\n".join(
                "8 " + " ".join(str(int(i)) for i in conn) for conn in elements
            )
        )
        lines.append(f"CELL_TYPES {n_elem}")
        lines.append("\n".join("12" for _ in range(n_elem)))

        # ---- POINT_DATA ----
        lines.append(f"POINT_DATA {n_nodes}")
        if self.displacement is not None:
            disp = np.asarray(self.displacement, dtype=float)
            lines.append("VECTORS displacement float")
            lines.append(_fmt_xyz(disp))

        def _point_scalar(name: str, arr) -> None:
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.shape[0] != n_nodes:
                return
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.append(_fmt(arr))

        if getattr(mesh, 'porosity', None) is not None:
            _point_scalar("porosity", mesh.porosity)
        if getattr(mesh, 'stiffness_reduction', None) is not None:
            _point_scalar("stiffness_reduction", mesh.stiffness_reduction)
        if getattr(mesh, 'ply_ids', None) is not None:
            _point_scalar("ply_id", mesh.ply_ids)

        # ---- CELL_DATA ----
        lines.append(f"CELL_DATA {n_elem}")

        def _cell_scalar(name: str, arr) -> None:
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.shape[0] != n_elem:
                return
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.append(_fmt(arr))

        _cell_scalar("von_mises", von_mises)
        for idx, comp in enumerate(
                ("sigma_xx", "sigma_yy", "sigma_zz",
                 "tau_yz", "tau_xz", "tau_xy")):
            _cell_scalar(comp, sig[:, idx])
        for idx, comp in enumerate(
                ("eps_xx", "eps_yy", "eps_zz",
                 "gamma_yz", "gamma_xz", "gamma_xy")):
            _cell_scalar(comp, eps[:, idx])

        if self.per_element_failure_index is not None:
            _cell_scalar("tsai_wu_index", self.per_element_failure_index)

        # Element-averaged nodal porosity over the 8 corner nodes.
        if getattr(mesh, 'porosity', None) is not None:
            vp_elem = np.mean(
                np.asarray(mesh.porosity, dtype=float)[elements], axis=1)
            _cell_scalar("Vp_elem", vp_elem)
        if getattr(mesh, 'elem_ply_ids', None) is not None:
            _cell_scalar("ply_id", mesh.elem_ply_ids)
        if getattr(mesh, 'ply_angles', None) is not None:
            _cell_scalar("ply_angle_deg", mesh.ply_angles)
        if getattr(mesh, 'void_elements', None) is not None:
            is_void = np.zeros(n_elem, dtype=float)
            void_idx = np.asarray(mesh.void_elements, dtype=np.int64).ravel()
            if void_idx.size:
                is_void[void_idx] = 1.0
            _cell_scalar("is_void", is_void)

        _cell_scalar(
            "knockdown",
            np.full(n_elem, float(self.knockdown), dtype=float))

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            f.write("\n")
        logger.info("Saved FE results (VTK): %s", filename)


class FESolver:
    """Linear static FE solver for porosity-degraded composite laminates.

    Workflow:
    1. Assemble K via GlobalAssembler
    2. Build BCs via BoundaryHandler
    3. Apply penalty method
    4. Solve K*u = F via spsolve
    5. Recover stresses at Gauss points
    6. Evaluate failure criterion at each GP (Tsai-Wu, Hashin, or max-stress)
    7. Compute knockdown factor

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh.
    material : MaterialProperties
        Material properties.
    porosity_field : PorosityField
        Porosity field for stiffness degradation.
    ply_angles : list of float or {'QI', 'UD'}, optional
        Optional list of ply angles (degrees), OR a string sentinel —
        ``'QI'`` (default, ``[0, 90, 45, -45]_s``) or ``'UD'`` (all-zero
        unidirectional). When provided, the resolved angle list is
        forwarded into the underlying :class:`CompositeMesh` per-element
        ``ply_angles`` array via :meth:`CompositeMesh.generate_mesh` only
        if the mesh has not yet been laid up with it — the mesh's
        ``ply_angles`` field remains the authoritative source for the
        per-element transformations used during solve. Passing ``None``
        is deprecated and resolved to ``'QI'`` with a
        :class:`DeprecationWarning` (#44 item 2).
    failure_criterion : {'tsai_wu', 'hashin', 'max_stress'}, optional
        Default failure criterion used by :meth:`solve` when no per-call
        override is supplied. ``'tsai_wu'`` (default) applies the
        quadratic Tsai-Wu interaction polynomial and preserves the
        historical bit-identical behavior. ``'hashin'`` uses the Hashin
        2D criterion with separate fiber/matrix tension/compression
        modes. ``'max_stress'`` uses an uncoupled maximum-stress check
        against each lamina strength. Validated against
        :attr:`SUPPORTED_FAILURE_CRITERIA`; an unknown value raises
        :class:`ValueError`.

    Notes
    -----
    ``ply_angles`` defaults — ``'QI'`` is the standardised default across
    :class:`EmpiricalSolver`, :class:`CompositeMesh`, and
    :class:`FESolver` (#44 item 2). The string sentinels expand to
    canonical baselines; explicit lists pass through unchanged. The
    constructor stores the resolved value on ``self.ply_angles`` so
    callers can introspect what layup the solver was built for; the
    actual per-element angles used during ``solve()`` come from
    ``self.mesh.ply_angles``, which is set when the mesh was constructed
    (passing ``ply_angles`` here does *not* relayup the mesh).
    """

    #: Supported failure criteria for :meth:`solve`. Used both at runtime
    #: (for validation) and as the documented enumeration.
    SUPPORTED_FAILURE_CRITERIA: Tuple[str, ...] = (
        'tsai_wu', 'hashin', 'max_stress')

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 porosity_field: PorosityField,
                 ply_angles: Optional[Union[List[float], str]] = 'QI',
                 failure_criterion: Literal[
                     'tsai_wu', 'hashin', 'max_stress'] = 'tsai_wu') -> None:
        self.mesh = mesh
        self.material = material
        self.porosity_field = porosity_field
        # Resolve the ply_angles sentinel (#44 item 2). The resolved value
        # is stored for introspection; the per-element angles consumed by
        # solve() come from ``mesh.ply_angles`` (set by the mesh's own
        # constructor). Previously ``ply_angles`` was stored-but-unused;
        # documenting the intentional decoupling here so the field has an
        # explicit contract.
        self.ply_angles = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='FESolver.ply_angles')
        self.assembler = GlobalAssembler(mesh, material, porosity_field)
        self.bc_handler = BoundaryHandler(mesh)
        if failure_criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {failure_criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )
        self.failure_criterion = failure_criterion

    def solve(self, loading: str = 'compression',
              applied_strain: float = -0.01,
              applied_load: float = -10.0,
              verbose: bool = False,
              failure_criterion: Optional[Literal[
                  'tsai_wu', 'hashin', 'max_stress']] = None,
              solver: Literal['direct', 'cg', 'minres'] = 'direct',
              rtol: float = 1e-9,
              diag_scale: bool = False,
              penalty_factor: float = 1e6) -> FieldResults:
        """Solve the static FE problem.

        Parameters
        ----------
        loading : str
            'compression', 'tension', 'shear', or 'ilss'.
        applied_strain : float
            Applied nominal strain (negative for compression). Used by the
            displacement-controlled modes ('compression', 'tension',
            'shear').
        applied_load : float
            Total midspan load (force) used by the force-controlled ILSS
            short-beam-shear mode (ASTM D2344). Ignored for the other
            modes.
        verbose : bool
            Print progress information.
        failure_criterion : {'tsai_wu', 'hashin', 'max_stress'}, optional
            Per-call override for the failure criterion. Defaults to the
            value passed to :meth:`__init__` (``'tsai_wu'`` if unset). When
            ``'tsai_wu'`` the result is bit-identical to the historical
            behavior; ``'hashin'`` and ``'max_stress'`` populate the
            per-mode breakdown on :class:`FieldResults`.
        solver : {'direct', 'cg', 'minres'}
            Linear solver to use for ``K u = F``. ``'direct'`` (default)
            uses :func:`scipy.sparse.linalg.spsolve` (sparse LU). For
            large meshes the LU fill-in dominates RAM; the penalty-modified
            matrix is SPD, so ``'cg'`` (conjugate gradient) with a Jacobi
            preconditioner is a memory-light alternative. ``'minres'``
            is offered for completeness when the matrix is symmetric but
            not strictly positive definite. Auto-switching is intentionally
            *not* performed — callers select the path explicitly
            (issue #57).
        rtol : float
            Relative-residual tolerance for the iterative solvers. Ignored
            when ``solver='direct'``.
        diag_scale : bool, optional
            If ``True``, symmetrically Jacobi-pre-scale the penalty-
            modified system before solving:
            ``(D^{-1/2} K_mod D^{-1/2}) y = D^{-1/2} F_mod``,
            ``u = D^{-1/2} y``, where ``D = diag(K_mod)``. The math is
            unchanged but the diagonal-conditioning ratio is reduced by
            2-3 decades on graded/voided meshes, which improves both LU
            backward error and CG/MINRES convergence. Defaults to
            ``False`` to preserve bit-identical legacy behavior; opt in
            when conditioning is a concern (issue #60).
        penalty_factor : float, optional
            Multiplier on ``max(diag(K))`` used by
            :meth:`BoundaryHandler.apply_penalty` to enforce Dirichlet
            BCs. Lowered from ``1e8`` to ``1e6`` (default) in issue #60
            to keep ``cond(K_mod)`` well below the float64 ceiling while
            still enforcing BCs to six decades. Tune higher only if BC
            slack is a problem; tune lower if conditioning is.

        Returns
        -------
        FieldResults
            Complete solution data.

        Raises
        ------
        ValueError
            If ``failure_criterion`` is not one of ``'tsai_wu'``,
            ``'hashin'``, ``'max_stress'`` (validated against
            :attr:`SUPPORTED_FAILURE_CRITERIA`), or if ``solver`` is not
            one of ``'direct'``, ``'cg'``, ``'minres'``.
        RuntimeError
            If the iterative solver fails to converge to ``rtol``, or if
            the direct solve produces non-finite values / a residual above
            ``1e-6``.
        """
        t0 = time.perf_counter()
        criterion = failure_criterion if failure_criterion is not None \
            else self.failure_criterion
        if criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )

        # 0. Mesh quality check
        check_mesh_quality(self.mesh, verbose=verbose)

        # 1. Assemble global stiffness
        if verbose:
            logger.info("Assembling global stiffness matrix...")
        K = self.assembler.assemble_stiffness(verbose=verbose)

        if verbose:
            t1 = time.perf_counter()
            logger.info("  Assembly time: %.2f s", t1 - t0)

        # 2. Build BCs
        if loading == 'compression':
            constrained, F = self.bc_handler.compression_bcs(applied_strain)
        elif loading == 'tension':
            constrained, F = self.bc_handler.tension_bcs(applied_strain)
        elif loading == 'shear':
            constrained, F = self.bc_handler.shear_bcs(applied_strain)
        elif loading == 'ilss':
            constrained, F = self.bc_handler.ilss_bcs(applied_load)
        else:
            raise ValueError(
                f"Unknown loading '{loading}'. "
                "Use compression/tension/shear/ilss."
            )

        if verbose:
            logger.info("  Applied %d displacement BCs", len(constrained))

        # 3. Apply penalty
        K_mod, F_mod = BoundaryHandler.apply_penalty(
            K, F, constrained, penalty_factor=penalty_factor,
        )

        # 3a. Conditioning diagnostic (issue #60). The diagonal ratio is
        # an inexpensive proxy for cond(K_mod) — full condest is O(n^2)
        # for sparse matrices and we want this on every solve. Warn the
        # user well before float64's ~1e16 headroom is exhausted.
        _diag = K_mod.diagonal()
        _diag_abs = np.abs(_diag)
        _diag_min = float(_diag_abs[_diag_abs > 0.0].min()) \
            if np.any(_diag_abs > 0.0) else 0.0
        _diag_max = float(_diag_abs.max()) if _diag_abs.size else 0.0
        cond_diag_ratio = (_diag_max / _diag_min) if _diag_min > 0.0 \
            else float('inf')
        logger.info(
            "Matrix conditioning: cond_diag_ratio=%.4e "
            "(penalty_factor=%.2e, diag_scale=%s)",
            cond_diag_ratio, penalty_factor, diag_scale,
        )
        if cond_diag_ratio > 1e12:
            logger.warning(
                "Matrix conditioning near float64 limit "
                "(cond_diag_ratio=%.2e); consider lowering "
                "penalty_factor or enabling diag_scale.",
                cond_diag_ratio,
            )

        # 4. Solve
        if solver not in ('direct', 'cg', 'minres'):
            raise ValueError(
                f"Unknown solver '{solver}'. "
                "Use 'direct', 'cg', or 'minres'."
            )
        if verbose:
            logger.info(
                "Solving system (%d DOFs) with solver='%s'...",
                self.mesh.n_dof, solver,
            )

        # 4a. Optional symmetric Jacobi pre-scaling (issue #60).
        # Replace (K_mod, F_mod) with (K_scaled, F_scaled) for the solve;
        # after solving, unscale y -> u via u = d_inv_sqrt * y.
        if diag_scale:
            _d = K_mod.diagonal()
            if not np.all(_d > 0):
                raise RuntimeError(
                    "Cannot apply diag_scale: K_mod has a non-positive "
                    "diagonal entry. Check assembly / penalty."
                )
            d_inv_sqrt = 1.0 / np.sqrt(_d)
            _D_is = scipy.sparse.diags(d_inv_sqrt)
            K_solve = (_D_is @ K_mod) @ _D_is
            F_solve = d_inv_sqrt * F_mod
            # Log the post-scaling diagonal ratio so the user can see
            # what the rescaling bought them.
            _d_scaled = K_solve.diagonal()
            _d_scaled_abs = np.abs(_d_scaled)
            _ds_min = float(_d_scaled_abs[_d_scaled_abs > 0.0].min()) \
                if np.any(_d_scaled_abs > 0.0) else 0.0
            _ds_max = float(_d_scaled_abs.max()) \
                if _d_scaled_abs.size else 0.0
            cond_diag_ratio_scaled = (_ds_max / _ds_min) \
                if _ds_min > 0.0 else float('inf')
            logger.info(
                "Matrix conditioning after diag_scale: "
                "cond_diag_ratio=%.4e (was %.4e)",
                cond_diag_ratio_scaled, cond_diag_ratio,
            )
        else:
            K_solve = K_mod
            F_solve = F_mod
            d_inv_sqrt = None

        if solver == 'direct':
            y = scipy.sparse.linalg.spsolve(K_solve, F_solve)

            # Hygiene checks on the solution vector
            if not np.isfinite(y).all():
                raise RuntimeError(
                    "spsolve produced non-finite values (NaN or Inf) in the solution "
                    "vector. Check matrix conditioning and boundary conditions."
                )
            _r = K_solve @ y - F_solve
            _rel_res = np.linalg.norm(_r) / max(np.linalg.norm(F_solve), 1.0)  # type: ignore[call-overload,operator]
            if _rel_res >= 1e-6:
                raise RuntimeError(
                    f"spsolve residual {_rel_res:.4e} exceeds tolerance 1e-6. "
                    "Check matrix conditioning or penalty factor."
                )
        else:
            # Jacobi (diagonal) preconditioner: K is SPD after penalty,
            # diag(K) is strictly positive.
            diag = K_solve.diagonal()
            if not np.all(diag > 0):
                raise RuntimeError(
                    "Cannot build Jacobi preconditioner: K_mod has a "
                    "non-positive diagonal entry. Check assembly / penalty."
                )
            M = scipy.sparse.diags(1.0 / diag)

            if solver == 'cg':
                y, info = scipy.sparse.linalg.cg(
                    K_solve, F_solve, M=M, rtol=rtol,
                )
            else:  # solver == 'minres'
                y, info = scipy.sparse.linalg.minres(
                    K_solve, F_solve, M=M, rtol=rtol,
                )

            _r = K_solve @ y - F_solve
            _norm_b = float(np.linalg.norm(F_solve))  # type: ignore[call-overload]
            _rel_res = float(
                np.linalg.norm(_r) / _norm_b if _norm_b > 0.0 else 0.0  # type: ignore[call-overload,operator]
            )
            # Compare the achieved relative residual against the user-
            # requested rtol directly. SciPy's iterative solvers can
            # report info=0 while still bouncing off the machine-
            # precision floor — if the user asked for sub-eps tolerance
            # they will (correctly) get a non-convergence error.
            _converged = info == 0 and _rel_res <= rtol * 10.0
            if not _converged:
                raise RuntimeError(
                    f"{solver} failed to converge: info={info}, "
                    f"achieved relative residual {_rel_res:.4e} "
                    f"(requested rtol={rtol:.4e})."
                )
            logger.info(
                "%s converged: relative residual %.4e (rtol=%.4e)",
                solver, _rel_res, rtol,
            )

        # 4b. Unscale if we Jacobi-pre-scaled. ``y`` solves the scaled
        # system; the physical displacement is ``u = D^{-1/2} y``.
        if diag_scale:
            u = d_inv_sqrt * y
        else:
            u = y

        if verbose:
            t2 = time.perf_counter()
            logger.info(
                "  Solve time: %.2f s, residual: %.4e", t2 - t1, _rel_res)
            t1 = t2

        # 5. Recover stresses and strains
        if verbose:
            logger.info("Recovering element stresses and strains...")

        n_elem = self.mesh.n_elements
        n_gp = 8  # 2x2x2

        stress_global = np.empty((n_elem, n_gp, 6))
        stress_local = np.empty((n_elem, n_gp, 6))
        strain_global = np.empty((n_elem, n_gp, 6))
        strain_local = np.empty((n_elem, n_gp, 6))

        for e in range(n_elem):
            if verbose and e % 500 == 0:
                logger.info(
                    "  Post-processing element %d/%d (%.1f%%)",
                    e, n_elem, 100.0 * e / n_elem,
                )

            dofs = self.assembler.element_dof_indices(e)
            u_elem = u[dofs]
            elem = self.assembler.create_element(e)

            sig_g = elem.stress_at_gauss_points(u_elem)
            eps_g = elem.strain_at_gauss_points(u_elem)

            stress_global[e] = sig_g
            strain_global[e] = eps_g

            # Transform to local coordinates. Stress uses T_sigma; engineering
            # strain (with gamma_ij = 2*eps_ij in slots 3-5) uses T_epsilon —
            # T_sigma applied to engineering strain leaves the shear components
            # off by 2x.
            ply_rad = np.radians(float(self.mesh.ply_angles[e]))
            T_sigma = stress_transformation_3d(ply_rad, axis='z')
            T_eps = strain_transformation_3d(ply_rad, axis='z')

            for g in range(n_gp):
                stress_local[e, g] = T_sigma @ sig_g[g]
                strain_local[e, g] = T_eps @ eps_g[g]

        # 6. Evaluate the selected failure criterion at each GP.
        #    per_elem_fi[e] is the max-over-GP failure index for element e
        #    (0.0 for skipped void elements); the scalar max_fi is its
        #    overall maximum. mode_indices captures the per-mode breakdown
        #    (NaN entries for Tsai-Wu, which does not separate modes).
        max_fi, per_elem_fi, mode_indices = self._evaluate_failure(
            stress_local, criterion=criterion)

        # 7. Compute knockdown as average-stress ratio (porous / pristine)
        # Both numerator and denominator use the same 3D FE framework so that
        # dimensional/mesh effects cancel.  For each element we compute what
        # the dominant stress component *would* be with pristine stiffness
        # at the same strain, then average. This avoids the CLT-vs-3D
        # mismatch that caused knockdown > 1.
        #
        # For ILSS short-beam shear the dominant component is tau_xz
        # (Voigt index 4); for the other modes it is sigma_xx (index 0).
        if loading == 'ilss':
            comp_idx = 4
        else:
            comp_idx = 0

        avg_sigma = np.mean(stress_global[:, :, comp_idx])

        # Pristine reference: compute the same Voigt component using the
        # rotated pristine stiffness applied to the recovered strain field.
        C_base = self.material.get_stiffness_matrix()
        pristine_sigma_sum = 0.0
        pristine_count = 0
        for e in range(n_elem):
            ply_rad = np.radians(float(self.mesh.ply_angles[e]))
            if abs(ply_rad) > 1e-15:
                C_prist_rot = rotate_stiffness_3d(C_base, ply_rad, axis='z')
            else:
                C_prist_rot = C_base
            for g in range(n_gp):
                eps = strain_global[e, g]
                pristine_sig = float(C_prist_rot[comp_idx, :] @ eps)
                pristine_sigma_sum += pristine_sig
                pristine_count += 1

        pristine_avg = pristine_sigma_sum / pristine_count if pristine_count > 0 else 1.0

        if abs(pristine_avg) > 1e-12:
            knockdown = abs(avg_sigma) / abs(pristine_avg)
        else:
            knockdown = 1.0
        knockdown = min(knockdown, 1.0)

        displacement = u.reshape(-1, 3)

        if verbose:
            t3 = time.perf_counter()
            logger.info("  Post-processing time: %.2f s", t3 - t1)
            logger.info("Total solve time: %.2f s", t3 - t0)
            logger.info("  Max %s FI: %.4f", criterion, max_fi)
            logger.info("  Knockdown factor: %.4f", knockdown)

        return FieldResults(
            displacement=displacement,
            stress_global=stress_global,
            stress_local=stress_local,
            strain_global=strain_global,
            strain_local=strain_local,
            max_failure_index=max_fi,
            knockdown=knockdown,
            per_element_failure_index=per_elem_fi,
            failure_criterion=criterion,
            failure_mode_indices=mode_indices,
        )

    #: Empty per-mode failure-index dict used when an element is skipped
    #: (void) or for criteria that do not populate a particular mode.
    _EMPTY_MODE_FI: Dict[str, float] = {
        'max_fi': 0.0,
        'fiber_t': 0.0,
        'fiber_c': 0.0,
        'matrix_t': 0.0,
        'matrix_c': 0.0,
        'shear': 0.0,
    }

    def _degraded_strengths(self, elem_Vp: float
                            ) -> Tuple[float, float, float, float, float, float]:
        """Return per-element porosity-degraded ply strengths.

        Implements the strength-degradation block shared by all failure
        criteria. Fiber-direction strengths (``Xt``, ``Xc``) follow the rule-
        of-mixtures fiber ratio (matrix porosity has only a weak indirect
        effect via ``E_m_eff``); transverse and shear strengths
        (``Yt``, ``Yc``, ``S12``, ``S23``) follow the Mori-Tanaka matrix
        stiffness ratio. Strengths are clamped to a small numerical floor so
        the per-criterion polynomial cannot divide by zero.

        Parameters
        ----------
        elem_Vp : float
            Element-average void volume fraction in [0, 1].

        Returns
        -------
        (Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s) : tuple of floats
            Floor-clamped degraded strengths in MPa.
        """
        mat = self.material
        C_m_pristine = mat.get_isotropic_matrix_stiffness()
        if elem_Vp > 1e-12:
            C_eff = _mt_effective_stiffness(
                C_m_pristine, elem_Vp,
                self.porosity_field.void_shape_radii,
                mat.matrix_poisson)
            # Matrix stiffness degradation ratio (matrix-dominated)
            r_matrix = np.sqrt(max(C_eff[0, 0] / C_m_pristine[0, 0], 0.0))
            # Fiber-direction ratio: scale by ROM ratio (much weaker effect)
            E_m = mat.matrix_modulus
            E_m_eff_approx = E_m * max(C_eff[0, 0] / C_m_pristine[0, 0], 0.0)
            Vf = mat.fiber_volume_fraction
            Vm = 1.0 - Vf
            r_fiber = (Vf * mat.fiber_modulus + Vm * E_m_eff_approx) / \
                      (Vf * mat.fiber_modulus + Vm * E_m)
            r_fiber = np.sqrt(max(r_fiber, 0.0))  # sqrt for strength vs stiffness
        else:
            r_matrix = 1.0
            r_fiber = 1.0

        Xt = mat.sigma_1t * r_fiber
        Xc = mat.sigma_1c * r_fiber
        Yt = mat.sigma_2t * r_matrix
        Yc = mat.sigma_2c * r_matrix
        S12 = mat.tau_12 * r_matrix
        S23 = mat.tau_ilss * r_matrix

        # Strengths approaching zero make the 1/X reciprocals overflow to
        # inf; clamp to a numerical floor so a heavily-degraded element
        # produces a large-but-finite failure index instead of poisoning the
        # global max with inf/NaN.
        strength_floor = 1e-3  # MPa
        return (max(Xt, strength_floor),
                max(Xc, strength_floor),
                max(Yt, strength_floor),
                max(Yc, strength_floor),
                max(S12, strength_floor),
                max(S23, strength_floor))

    def _evaluate_failure(self, stress_local: np.ndarray,
                          criterion: str = 'tsai_wu'
                          ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """Evaluate the chosen failure criterion at every Gauss point.

        Dispatches to :meth:`_evaluate_tsai_wu`, :meth:`_evaluate_hashin`,
        or :meth:`_evaluate_max_stress` element by element. Per-element
        strength degradation is computed once via :meth:`_degraded_strengths`
        and reused by the per-criterion polynomials.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape (n_elem, n_gp, 6) local stresses.
        criterion : {'tsai_wu', 'hashin', 'max_stress'}
            Failure criterion to apply.

        Returns
        -------
        max_fi : float
            Overall maximum failure index.
        per_elem_fi : np.ndarray
            Shape (n_elem,) max-over-Gauss-point failure index per element
            (0.0 for skipped void elements).
        mode_indices : dict
            Per-mode breakdown at the element/GP where ``max_fi`` is
            attained. Keys: ``'max_fi'``, ``'fiber_t'``, ``'fiber_c'``,
            ``'matrix_t'``, ``'matrix_c'``, ``'shear'``. For Tsai-Wu the
            per-mode entries are ``NaN`` (the coupled polynomial does not
            separate modes).
        """
        if criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure criterion {criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )

        n_elem, _, _ = stress_local.shape
        per_elem_fi = np.zeros(n_elem, dtype=float)
        # Defense in depth: non-finite porosity silently corrupts elem_Vp.
        if not np.all(np.isfinite(self.mesh.porosity)):  # type: ignore[call-overload]
            raise ValueError(
                f"mesh.porosity contains non-finite values; refusing to evaluate "
                f"{criterion} on a corrupted porosity field."
            )

        # #114: hoist the per-element mean computation outside the loop.
        # The old `np.mean(porosity[elements[e]])` per iteration was an
        # O(n_elem) Python loop where O(1) vectorized NumPy works; this gives
        # ~143x on the inner step and ~1-2 s on a typical 5x5 sweep.
        elem_Vp_all = np.clip(
            np.mean(self.mesh.porosity[self.mesh.elements], axis=1),
            0.0, 1.0,
        )

        max_fi = 0.0
        best_mode_indices: Dict[str, float] = dict(self._EMPTY_MODE_FI)
        if criterion == 'tsai_wu':
            # Tsai-Wu polynomial couples all components — per-mode breakdown
            # is undefined. Surface NaN sentinels so downstream consumers see
            # "criterion did not separate modes" rather than spurious zeros.
            best_mode_indices = {
                'max_fi': 0.0,
                'fiber_t': float('nan'),
                'fiber_c': float('nan'),
                'matrix_t': float('nan'),
                'matrix_c': float('nan'),
                'shear': float('nan'),
            }

        for e in range(n_elem):
            elem_Vp = float(elem_Vp_all[e])

            # Skip void elements (carry no meaningful load)
            if elem_Vp > 0.95:
                continue

            strengths = self._degraded_strengths(elem_Vp)
            s_all = stress_local[e]  # (n_gp, 6)

            if criterion == 'tsai_wu':
                fi_per_gp = self._evaluate_tsai_wu(s_all, strengths, e, elem_Vp)
                elem_max = float(fi_per_gp.max())
                per_elem_fi[e] = elem_max
                if elem_max > max_fi:
                    max_fi = elem_max
                    best_mode_indices['max_fi'] = elem_max
            else:
                if criterion == 'hashin':
                    mode_fi_per_gp = self._evaluate_hashin(s_all, strengths)
                else:  # 'max_stress'
                    mode_fi_per_gp = self._evaluate_max_stress(s_all, strengths)
                # mode_fi_per_gp is a dict of arrays, each shape (n_gp,).
                fi_per_gp = mode_fi_per_gp['max_fi']
                if not np.all(np.isfinite(fi_per_gp)):
                    bad_g = int(np.argmax(~np.isfinite(fi_per_gp)))
                    raise ValueError(
                        f"{criterion} failure index is non-finite at element "
                        f"{e}, Gauss point {bad_g} (Vp={elem_Vp:.4f}, "
                        f"stress={s_all[bad_g].tolist()})."
                    )
                elem_max = float(fi_per_gp.max())
                per_elem_fi[e] = elem_max
                if elem_max > max_fi:
                    max_fi = elem_max
                    g_max = int(np.argmax(fi_per_gp))
                    best_mode_indices = {
                        'max_fi': elem_max,
                        'fiber_t': float(mode_fi_per_gp['fiber_t'][g_max]),
                        'fiber_c': float(mode_fi_per_gp['fiber_c'][g_max]),
                        'matrix_t': float(mode_fi_per_gp['matrix_t'][g_max]),
                        'matrix_c': float(mode_fi_per_gp['matrix_c'][g_max]),
                        'shear': float(mode_fi_per_gp['shear'][g_max]),
                    }

        return float(max_fi), per_elem_fi, best_mode_indices

    def _evaluate_tsai_wu(self, s_all: np.ndarray,
                          strengths: Tuple[float, float, float, float, float, float],
                          e: int, elem_Vp: float) -> np.ndarray:
        """Tsai-Wu polynomial evaluated for one element's Gauss points.

        Bit-identical to the historical implementation. Returns the per-GP
        failure index array (shape ``(n_gp,)``).
        """
        Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            F1 = 1.0 / Xt_s - 1.0 / Xc_s
            F2 = 1.0 / Yt_s - 1.0 / Yc_s
            F3 = F2
            F11 = 1.0 / (Xt_s * Xc_s)
            F22 = 1.0 / (Yt_s * Yc_s)
            F33 = F22
            F44 = 1.0 / S23_s**2
            F55 = 1.0 / S12_s**2
            F66 = 1.0 / S12_s**2
            # F12, F23 use sqrt of a product. Guard against negative
            # products in case future refactors break the F11/F22/F33 sign.
            F11_F22 = max(F11 * F22, 0.0)
            F22_F33 = max(F22 * F33, 0.0)
            F12 = -0.5 * np.sqrt(F11_F22)
            F13 = F12
            F23 = -0.5 * np.sqrt(F22_F33)

        # Vectorize across all Gauss points of this element (#41).
        fi_per_gp = (
            F1 * s_all[:, 0] + F2 * s_all[:, 1] + F3 * s_all[:, 2]
            + F11 * s_all[:, 0]**2 + F22 * s_all[:, 1]**2 + F33 * s_all[:, 2]**2
            + F44 * s_all[:, 3]**2 + F55 * s_all[:, 4]**2 + F66 * s_all[:, 5]**2
            + 2 * F12 * s_all[:, 0] * s_all[:, 1]
            + 2 * F13 * s_all[:, 0] * s_all[:, 2]
            + 2 * F23 * s_all[:, 1] * s_all[:, 2]
        )
        if not np.all(np.isfinite(fi_per_gp)):
            bad_g = int(np.argmax(~np.isfinite(fi_per_gp)))
            raise ValueError(
                f"Tsai-Wu failure index is non-finite at element {e}, "
                f"Gauss point {bad_g} (Vp={elem_Vp:.4f}, "
                f"stress={s_all[bad_g].tolist()}). This usually indicates "
                f"a degenerate stiffness or strength matrix; refine the "
                f"mesh or check input bounds."
            )
        return fi_per_gp

    def _evaluate_hashin(self, s_all: np.ndarray,
                         strengths: Tuple[float, float, float, float, float, float]
                         ) -> Dict[str, np.ndarray]:
        """2D Hashin failure indices for unidirectional plies.

        Implements the standard four-mode Hashin criterion (Hashin, 1980).
        Indices are computed per Gauss point on the in-plane local stresses
        ``(σ_11, σ_22, τ_12)``; ``σ_33`` and out-of-plane shears are ignored
        because the standard formulation is 2D. The ``shear`` slot returns
        the in-plane ``(τ_12 / S_12)^2`` contribution for completeness.

        Returns a dict of per-GP arrays:
        ``{'max_fi', 'fiber_t', 'fiber_c', 'matrix_t', 'matrix_c', 'shear'}``.
        """
        Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
        sigma_11 = s_all[:, 0]
        sigma_22 = s_all[:, 1]
        tau_12 = s_all[:, 5]

        # Fiber tension: σ_11 >= 0
        ft = (sigma_11 / Xt_s) ** 2 + (tau_12 / S12_s) ** 2
        ft = np.where(sigma_11 >= 0.0, ft, 0.0)

        # Fiber compression: σ_11 < 0
        fc = (sigma_11 / Xc_s) ** 2
        fc = np.where(sigma_11 < 0.0, fc, 0.0)

        # Matrix tension: σ_22 >= 0
        mt = (sigma_22 / Yt_s) ** 2 + (tau_12 / S12_s) ** 2
        mt = np.where(sigma_22 >= 0.0, mt, 0.0)

        # Matrix compression: σ_22 < 0
        mc_term = ((Yc_s / (2.0 * S23_s)) ** 2 - 1.0) * (sigma_22 / Yc_s)
        mc = (sigma_22 / (2.0 * S23_s)) ** 2 + mc_term + (tau_12 / S12_s) ** 2
        mc = np.where(sigma_22 < 0.0, mc, 0.0)

        shear = (tau_12 / S12_s) ** 2

        max_fi = np.maximum.reduce([ft, fc, mt, mc])
        return {
            'max_fi': max_fi,
            'fiber_t': ft,
            'fiber_c': fc,
            'matrix_t': mt,
            'matrix_c': mc,
            'shear': shear,
        }

    def _evaluate_max_stress(self, s_all: np.ndarray,
                             strengths: Tuple[float, float, float, float, float, float]
                             ) -> Dict[str, np.ndarray]:
        """Maximum-stress failure indices.

        ``FI_i = |σ_i| / X_i_allowable`` per component (signed split for
        normals: tensile vs compressive allowable). The reported ``max_fi``
        is the maximum across all five mode/component buckets. Returns the
        same per-GP dict shape as :meth:`_evaluate_hashin`; unused entries
        are zeroed (rather than NaN) since each mode is well-defined for
        max-stress.
        """
        Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
        sigma_11 = s_all[:, 0]
        sigma_22 = s_all[:, 1]
        sigma_33 = s_all[:, 2]
        tau_23 = s_all[:, 3]
        tau_13 = s_all[:, 4]
        tau_12 = s_all[:, 5]

        ft = np.where(sigma_11 >= 0.0, sigma_11 / Xt_s, 0.0)
        fc = np.where(sigma_11 < 0.0, -sigma_11 / Xc_s, 0.0)
        # Matrix uses worst of σ_22 and σ_33 (transverse normals share the
        # same in-plane transverse strength).
        mt_22 = np.where(sigma_22 >= 0.0, sigma_22 / Yt_s, 0.0)
        mt_33 = np.where(sigma_33 >= 0.0, sigma_33 / Yt_s, 0.0)
        mt = np.maximum(mt_22, mt_33)
        mc_22 = np.where(sigma_22 < 0.0, -sigma_22 / Yc_s, 0.0)
        mc_33 = np.where(sigma_33 < 0.0, -sigma_33 / Yc_s, 0.0)
        mc = np.maximum(mc_22, mc_33)
        # Shear: worst of all three engineering shear components against the
        # appropriate allowable (S_23 for the 23 plane, S_12 for 12 / 13).
        shear = np.maximum.reduce([
            np.abs(tau_12) / S12_s,
            np.abs(tau_13) / S12_s,
            np.abs(tau_23) / S23_s,
        ])

        max_fi = np.maximum.reduce([ft, fc, mt, mc, shear])
        return {
            'max_fi': max_fi,
            'fiber_t': ft,
            'fiber_c': fc,
            'matrix_t': mt,
            'matrix_c': mc,
            'shear': shear,
        }

    @staticmethod
    def export_results(field_results: 'FieldResults', filename: str,
                       fmt: str = 'json',
                       mesh: Optional['CompositeMesh'] = None,
                       include_raw: bool = False) -> None:
        """Export FE results to a JSON summary or a VTK field file.

        With ``fmt='json'`` (the default, unchanged legacy behavior) this
        saves displacement statistics, stress/strain summaries, failure data,
        and knockdown factor; large arrays are summarized (min/max/mean/std)
        rather than stored in full.

        With ``fmt='vtk'`` it delegates to :meth:`FieldResults.to_vtk` and
        writes the full hex mesh plus per-element fields as a legacy ASCII
        ``UNSTRUCTURED_GRID`` for ParaView / VisIt / PyVista. The richer
        per-element/per-node API lives on ``FieldResults.to_vtk`` directly;
        this ``fmt='vtk'`` path is a convenience shim for callers that
        already hold an ``FESolver``.

        Parameters
        ----------
        field_results : FieldResults
            Results from FESolver.solve().
        filename : str
            Output file path (``.json`` or ``.vtk``).
        fmt : str
            ``'json'`` (default) or ``'vtk'``.
        mesh : CompositeMesh, optional
            Required when ``fmt='vtk'`` (supplies geometry/connectivity).
        include_raw : bool
            When ``True`` (and ``fmt='json'``), also write a sidecar
            ``<filename>.npz`` containing the raw displacement/stress/strain
            arrays so a full audit can re-derive the per-key summary
            statistics. Default ``False`` so existing outputs are not
            bloated (#55).
        """
        fmt = str(fmt).lower()
        if fmt == 'vtk':
            if mesh is None:
                raise ValueError(
                    "export_results(fmt='vtk') requires the `mesh` argument "
                    "(pass the CompositeMesh used by the solver)."
                )
            field_results.to_vtk(mesh, filename)
            return
        if fmt != 'json':
            raise ValueError(
                f"Unknown export format {fmt!r}. Use 'json' or 'vtk'."
            )

        def _array_stats(arr: np.ndarray) -> dict:
            """Compute summary statistics for an array."""
            return {
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
            }

        results_data = {
            'displacement': {
                'n_nodes': int(field_results.displacement.shape[0]),
                'ux': _array_stats(field_results.displacement[:, 0]),
                'uy': _array_stats(field_results.displacement[:, 1]),
                'uz': _array_stats(field_results.displacement[:, 2]),
            },
            'stress_global': {
                'n_elements': int(field_results.stress_global.shape[0]),
                'n_gauss_points': int(field_results.stress_global.shape[1]),
                'sigma_11': _array_stats(field_results.stress_global[:, :, 0]),
                'sigma_22': _array_stats(field_results.stress_global[:, :, 1]),
                'sigma_33': _array_stats(field_results.stress_global[:, :, 2]),
                'tau_23': _array_stats(field_results.stress_global[:, :, 3]),
                'tau_13': _array_stats(field_results.stress_global[:, :, 4]),
                'tau_12': _array_stats(field_results.stress_global[:, :, 5]),
            },
            'stress_local': {
                'sigma_11': _array_stats(field_results.stress_local[:, :, 0]),
                'sigma_22': _array_stats(field_results.stress_local[:, :, 1]),
                'tau_12': _array_stats(field_results.stress_local[:, :, 5]),
            },
            'strain_global': {
                'eps_11': _array_stats(field_results.strain_global[:, :, 0]),
                'eps_22': _array_stats(field_results.strain_global[:, :, 1]),
                'gamma_12': _array_stats(field_results.strain_global[:, :, 5]),
            },
            'failure': {
                'max_tsai_wu_index': float(field_results.max_failure_index),
                'max_failure_index': float(field_results.max_failure_index),
                'criterion': str(getattr(field_results,
                                          'failure_criterion', 'tsai_wu')),
                'mode_indices': (
                    {k: float(v) for k, v in field_results.failure_mode_indices.items()}
                    if field_results.failure_mode_indices is not None
                    else None
                ),
                'knockdown_factor': float(field_results.knockdown),
            },
        }

        output = {
            'schema_version': JSON_SCHEMA_VERSION,
            'format': FORMAT_FE_FIELDS,
            'provenance': _build_provenance(),
            **results_data,
        }
        if include_raw:
            # Sidecar file path lives next to the JSON so users see them
            # together; ``np.savez`` will append ``.npz`` if missing.
            npz_path = f"{filename}.npz"
            arrays = {
                'displacement': np.asarray(field_results.displacement),
                'stress_global': np.asarray(field_results.stress_global),
                'stress_local': np.asarray(field_results.stress_local),
                'strain_global': np.asarray(field_results.strain_global),
                'strain_local': np.asarray(field_results.strain_local),
            }
            if field_results.per_element_failure_index is not None:
                arrays['per_element_failure_index'] = np.asarray(
                    field_results.per_element_failure_index)
            np.savez(npz_path, **arrays)
            output['raw_sidecar'] = os.path.basename(npz_path)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=_json_default)
        logger.info("Saved FE results: %s", filename)

