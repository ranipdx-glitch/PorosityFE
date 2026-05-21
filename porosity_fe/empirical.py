"""Empirical (analytical) porosity-strength knockdown solver."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ._ply_angles import _resolve_ply_angles
from .fatigue import FatigueModel
from .materials import MaterialProperties
from .mesh import CompositeMesh
from .results import FailureResult

logger = logging.getLogger("porosity_fe_analysis")

class EmpiricalSolver:
    """Fast analytical solver using empirical porosity-strength models.

    Coefficients are calibrated against quasi-isotropic data and scaled by
    a layup-dependent matrix-dominated fraction so that fiber-dominated
    layups (e.g. UD [0]_n) see a smaller porosity penalty than QI layups.

    Knockdowns are evaluated at the specimen-average porosity (Vp_mean),
    matching how the original correlations were calibrated — not at the
    local peak Vp that clustered distributions produce.
    """

    # QI-calibrated coefficients (Elhajjar 2025, Sci. Rep. 15:25977).
    # `_F_MD_REF = 0.5` below is the LAYUP-SCALING reference (scale = 1.0 at
    # f_md = 0.5), NOT a property of the Elhajjar coupon layup itself
    # (`[0/45/90/-45/0]_s`, which the binning rule below puts at f_md = 0.4).
    # The coefficients were tuned with the layup-scaling already applied,
    # so they represent the model's effective f_md = 0.5 baseline rather
    # than the raw fit on a single layup.
    # See README "Empirical Strength Knockdown" for definitions, units (alpha, n
    # are dimensionless when Vp is a fraction in [0, 1]), validity bounds, and
    # the calibration recipe for custom materials.
    # Modes: 'compression' (sigma_1c, fiber+matrix), 'tension' (sigma_1t,
    # fiber-dominated), 'shear' (tau_12, in-plane, matrix-dominated),
    # 'ilss' (tau_ilss, short-beam, matrix/interface-dominated),
    # 'transverse_tension' (sigma_2t, in-plane transverse, matrix-dominated;
    # alpha matched to ilss because both fail by matrix/interface-dominated
    # mechanisms — see issue #35).
    _JUDD_WRIGHT_ALPHA_QI = {
        'compression': 6.9, 'tension': 3.9, 'shear': 8.0, 'ilss': 10.0,
        'transverse_tension': 10.0,
    }
    _POWER_LAW_N_QI = {
        'compression': 2.8, 'tension': 1.8, 'shear': 3.5, 'ilss': 4.5,
        'transverse_tension': 4.5,
    }
    _LINEAR_BETA_QI = {
        'compression': 5.5, 'tension': 3.5, 'shear': 7.0, 'ilss': 9.0,
        'transverse_tension': 9.0,
    }
    PRISTINE_STRENGTH_KEY = {
        'compression': 'sigma_1c', 'tension': 'sigma_1t',
        'shear': 'tau_12', 'ilss': 'tau_ilss',
        'transverse_tension': 'sigma_2t',
    }
    # Modes whose porosity sensitivity is matrix-/interface-dominated even in
    # UD layups (where the longitudinal-fiber metric would otherwise drive
    # f_md to ~0).  These modes use the elevated ``_F_MD_FLOOR_ILSS`` floor.
    _MATRIX_DOMINATED_MODES = frozenset({'ilss', 'transverse_tension'})
    # QI reference fraction and minimum floor
    _F_MD_REF = 0.5    # f_md for the QI layup used in calibration
    _F_MD_FLOOR = 0.15  # even UD has some matrix sensitivity
    _F_MD_FLOOR_ILSS = 0.80  # ILSS / transverse-tension are always matrix-dominated

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 ply_angles: Optional[Union[List[float], str]] = 'QI',
                 *,
                 judd_wright_alpha: Optional[Dict[str, float]] = None,
                 power_law_n: Optional[Dict[str, float]] = None,
                 linear_beta: Optional[Dict[str, float]] = None):
        """Empirical knockdown solver.

        Parameters
        ----------
        mesh : CompositeMesh
            Mesh whose nodal porosity drives the knockdown.
        material : MaterialProperties
            Composite material; supplies pristine strengths.
        ply_angles : list of float or {'QI', 'UD'}, optional
            Per-ply orientation in degrees, OR a string sentinel:

            - ``'QI'`` (default) -> ``[0, 90, 45, -45]_s`` quasi-isotropic
              baseline (``f_md = 0.5``, matches the calibration basis).
            - ``'UD'`` -> ``[0, 0, 0, 0]`` unidirectional baseline.
            - explicit list of floats -> used verbatim.

            Passing ``None`` is deprecated and currently resolved to
            ``'QI'`` with a :class:`DeprecationWarning` (#44 item 2);
            this back-compat path will be removed in a future major
            version. ``judd_wright_alpha`` / ``power_law_n`` /
            ``linear_beta`` are optional partial overrides for the
            QI-calibrated coefficients (see README "Empirical Strength
            Knockdown"). Each accepts a dict keyed by mode
            (``'compression'`` / ``'tension'`` / ``'shear'`` / ``'ilss'``);
            modes that are absent fall back to the QI defaults. Override
            values are layup-scaled exactly like the defaults: at
            ``f_md = 0.5`` the scale is 1.0, so a passed-in ``alpha`` is
            the value used directly.

        Notes
        -----
        ``ply_angles`` defaults — ``'QI'`` is the standardised default
        across :class:`EmpiricalSolver`, :class:`CompositeMesh`, and
        :class:`FESolver` (#44 item 2). The string sentinels expand to
        canonical baselines; explicit lists pass through unchanged.
        """
        self.mesh = mesh
        self.material = material
        self.nodal_knockdown = None

        # Resolve the ply_angles sentinel (#44 item 2). ``None`` is the
        # deprecated path and emits a DeprecationWarning inside
        # ``_resolve_ply_angles``.
        ply_angles_resolved = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='EmpiricalSolver.ply_angles')

        # Resolve coefficient dicts: per-mode merge of class default with override.
        alpha_qi = self._merge_coefficient_override(
            self._JUDD_WRIGHT_ALPHA_QI, judd_wright_alpha, 'judd_wright_alpha')
        n_qi = self._merge_coefficient_override(
            self._POWER_LAW_N_QI, power_law_n, 'power_law_n')
        beta_qi = self._merge_coefficient_override(
            self._LINEAR_BETA_QI, linear_beta, 'linear_beta')

        # Compute layup-dependent scaling
        self.f_md = self._matrix_dominated_fraction(ply_angles_resolved)

        # Build scaled coefficient dicts. Explicit annotations let static
        # checkers narrow `self.JUDD_WRIGHT_ALPHA[mode]` etc. to `float`
        # at the vectorized call sites in `apply_loading` (#114/#115).
        self.JUDD_WRIGHT_ALPHA: Dict[str, float] = {}
        self.POWER_LAW_N: Dict[str, float] = {}
        self.LINEAR_BETA: Dict[str, float] = {}
        for mode in ['compression', 'tension', 'shear', 'ilss',
                     'transverse_tension']:
            s = self._layup_scale(mode)
            self.JUDD_WRIGHT_ALPHA[mode] = alpha_qi[mode] * s
            self.POWER_LAW_N[mode] = max(n_qi[mode] * s, 0.1)
            self.LINEAR_BETA[mode] = beta_qi[mode] * s

    @classmethod
    def _merge_coefficient_override(cls, defaults: Dict[str, float],
                                    override: Optional[Dict[str, float]],
                                    name: str) -> Dict[str, float]:
        """Validate ``override`` and merge it onto ``defaults``."""
        if override is None:
            return dict(defaults)
        if not isinstance(override, dict):
            raise TypeError(
                f"{name} must be a dict mapping mode -> coefficient, "
                f"got {type(override).__name__}."
            )
        valid_modes = set(cls.PRISTINE_STRENGTH_KEY)
        unknown = set(override) - valid_modes
        if unknown:
            raise ValueError(
                f"{name} has unknown mode keys {sorted(unknown)}. "
                f"Use a subset of {sorted(valid_modes)}."
            )
        for mode, value in override.items():
            if not isinstance(value, (int, float, np.floating, np.integer)):
                raise TypeError(
                    f"{name}[{mode!r}] must be a number, "
                    f"got {type(value).__name__}."
                )
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name}[{mode!r}] must be a positive finite number, got {value!r}."
                )
        return {**defaults, **override}

    @staticmethod
    def _matrix_dominated_fraction(ply_angles: Optional[List[float]]) -> float:
        """Fraction of matrix-dominated plies in the layup (0 to 1).

        - 0-degree plies contribute 0 (fiber-dominated)
        - +/-45-degree plies contribute 0.5 (intermediate)
        - 90-degree plies contribute 1.0 (matrix-dominated)

        Returns 0.5 (QI reference) if ply_angles is None.
        """
        if ply_angles is None or len(ply_angles) == 0:
            return 0.5  # default = QI reference
        total = 0.0
        for angle in ply_angles:
            a = abs(angle) % 180
            if a <= 10:        # near 0°
                total += 0.0
            elif a >= 80:      # near 90°
                total += 1.0
            else:              # off-axis (30°, 45°, 60°, etc.)
                total += 0.5
        return total / len(ply_angles)

    # TODO(#140): The linear f_md / _F_MD_REF scaling is preserved here for
    # historical compatibility. Investigation in #140 measured a relative
    # error of up to 33.5% against a CLT-derived stiffness-retention proxy
    # (sqrt(Ex_layup(Vp)/Ex_layup(0)) / sqrt(Ex_QI(Vp)/Ex_QI(0)) over
    # Vp in [0.005, 0.05]) for a UD [0,0,0]_s layup; >5% error also seen
    # on UD-heavy [0_2,90]_s and off-axis [0,15,-15]_s layups. A
    # polynomial or interpolated lookup should be evaluated against an
    # independent reference dataset (FE simulation or experimental
    # coupons spanning the intermediate f_md range) before the scaling
    # is replaced. See TestLayupScaleRegressionPin in
    # tests/test_porosity_fe.py.
    def _layup_scale(self, mode: str) -> float:
        """Scaling factor for empirical coefficients based on layup.

        Maps f_md to a coefficient multiplier:
        - f_md = f_md_ref (0.5, QI) -> scale = 1.0 (unchanged)
        - f_md = 0 (UD) -> scale = floor (0.15 for most modes, 0.80 for ILSS)
        - f_md > f_md_ref -> scale > 1.0 (more matrix-dominated than QI)
        """
        floor = (self._F_MD_FLOOR_ILSS if mode in self._MATRIX_DOMINATED_MODES
                 else self._F_MD_FLOOR)
        ref = self._F_MD_REF
        if ref < 1e-12:
            return 1.0
        raw = self.f_md / ref
        return max(raw, floor)

    @staticmethod
    def _check_internal_Vp(Vp: float) -> float:
        # Defensive: tolerate fp overshoot (~1e-15) from element-mean averaging
        # by clipping to [0, 1]; reject non-finite outright.
        if not np.isfinite(Vp):
            raise ValueError(f"Internal Vp is non-finite: {Vp!r}")
        return float(np.clip(Vp, 0.0, 1.0))

    @staticmethod
    def _validate_user_kd_callable(model_func: Callable[[float, str], float],
                                   mode: str,
                                   max_Vp: float = 1.0,
                                   n_grid: int = 11) -> None:
        """Validate that a user-supplied knockdown callable is well-behaved.

        The callable contract is ``model(Vp: float, mode: str) -> float in
        [0, 1]``. We sample ``Vp`` on a uniform grid over ``[0, max_Vp]`` and
        check that the return is finite and in the closed unit interval.
        Reuses :meth:`_check_internal_Vp` so the same overshoot/finite policy
        applies as the built-in models.

        Raises
        ------
        TypeError
            If ``model_func`` is not callable.
        ValueError
            If the callable returns a non-finite value, or a value outside
            ``[0, 1]``, on any grid point.
        """
        if not callable(model_func):
            raise TypeError(
                f"User knockdown model must be callable; got "
                f"{type(model_func).__name__}."
            )
        grid = np.linspace(0.0, float(max_Vp), int(n_grid))
        for Vp in grid:
            Vp = EmpiricalSolver._check_internal_Vp(float(Vp))
            try:
                kd = model_func(Vp, mode)
            except Exception as exc:
                raise ValueError(
                    f"User knockdown model raised {type(exc).__name__} at "
                    f"Vp={Vp:.3f}, mode={mode!r}: {exc}"
                ) from exc
            kd_f = float(kd)
            if not np.isfinite(kd_f):
                raise ValueError(
                    f"User knockdown model returned non-finite value "
                    f"{kd_f!r} at Vp={Vp:.3f}, mode={mode!r}; "
                    f"expected a finite float in [0, 1]."
                )
            if kd_f < 0.0 or kd_f > 1.0:
                raise ValueError(
                    f"User knockdown model returned {kd_f!r} at Vp={Vp:.3f}, "
                    f"mode={mode!r}; expected a value in [0, 1]."
                )

    def _judd_wright(self, Vp: float, mode: str) -> float:
        """Judd-Wright knockdown: KD = exp(-alpha * Vp).

        Vp is a void volume fraction in [0, 1]. ``alpha`` is the
        layup-scaled, mode-specific sensitivity coefficient (see
        ``JUDD_WRIGHT_ALPHA`` and the README "Empirical Strength
        Knockdown" section for definitions and ranges).
        """
        Vp = self._check_internal_Vp(Vp)
        alpha = self.JUDD_WRIGHT_ALPHA[mode]
        return float(np.exp(-alpha * Vp))

    def _power_law(self, Vp: float, mode: str) -> float:
        """Power-law knockdown: KD = (1 - Vp)**n.

        Vp is a void volume fraction in [0, 1]. ``n`` is the
        layup-scaled, mode-specific exponent (see ``POWER_LAW_N`` and
        the README "Empirical Strength Knockdown" section for
        definitions and ranges).
        """
        Vp = self._check_internal_Vp(Vp)
        n = self.POWER_LAW_N[mode]
        return float((1.0 - Vp)**n)

    def _linear(self, Vp: float, mode: str) -> float:
        Vp = self._check_internal_Vp(Vp)
        beta = self.LINEAR_BETA[mode]
        return float(max(1.0 - beta * Vp, 0.0))

    def _get_pristine_strength(self, mode: str) -> float:
        if mode not in self.PRISTINE_STRENGTH_KEY:
            raise ValueError(
                f"Unknown loading mode {mode!r}. "
                f"Use one of {sorted(self.PRISTINE_STRENGTH_KEY)}."
            )
        return getattr(self.material, self.PRISTINE_STRENGTH_KEY[mode])

    def _environment_knockdown_factor(self, mode: str,
                                      environment: Optional[Dict[str, float]]
                                      ) -> float:
        """Resolve the hygrothermal knockdown factor (issue #59).

        Returns ``1.0`` (back-compat no-op) when ``environment`` is
        ``None``. Otherwise delegates to
        :meth:`MaterialProperties.environment_knockdown`, pulling ``T``
        and ``M`` out of the dict (either key is optional — a missing
        key falls back to the material's ``T_service`` / ``M_service``).
        """
        if environment is None:
            return 1.0
        if not isinstance(environment, dict):
            raise TypeError(
                f"environment must be a dict (e.g. {{'T': 80.0, 'M': 1.2}}), "
                f"got {type(environment).__name__}."
            )
        allowed_keys = {'T', 'M'}
        unknown = set(environment) - allowed_keys
        if unknown:
            raise ValueError(
                f"environment has unknown keys {sorted(unknown)}. "
                f"Use a subset of {sorted(allowed_keys)}."
            )
        T = environment.get('T')
        M = environment.get('M')
        return float(self.material.environment_knockdown(mode, T=T, M=M))

    def _fatigue_knockdown_factor(self, mode: str,
                                  cycles: Optional[float],
                                  R: Optional[float]) -> float:
        """Resolve the S-N fatigue knockdown factor (issue #59).

        Returns ``1.0`` (back-compat no-op) when ``cycles`` is ``None``.
        Otherwise instantiates a default :class:`FatigueModel` and
        evaluates :meth:`FatigueModel.knockdown_factor` at the requested
        ``cycles`` and ``R`` (``R = 0.1`` if not specified).
        """
        if cycles is None:
            return 1.0
        R_eff = 0.1 if R is None else float(R)
        return float(FatigueModel().knockdown_factor(mode, cycles, R_eff))

    def _apply_discrete_void_scf(self, base_knockdown: np.ndarray, mode: str) -> np.ndarray:
        kd = base_knockdown.copy()
        for void in self.mesh.porosity_field.discrete_voids:
            scf_dict = void.stress_concentration_factor()
            scf = scf_dict.get(mode, 1.0)
            dist = void.distance_field(self.mesh.nodes[:, 0],
                                        self.mesh.nodes[:, 1],
                                        self.mesh.nodes[:, 2])
            influence = np.exp(-np.maximum(dist, 0) / max(void.radii))
            kd *= (1.0 - influence * (1.0 - 1.0 / scf))
        return kd

    def _resolve_knockdown_model(
            self, model: Union[str, Callable[[float, str], float]], mode: str
    ) -> Tuple[Callable[[float, str], float], bool]:
        """Resolve ``model`` to a ``(callable, is_user_supplied)`` pair.

        Built-in string names dispatch to the corresponding ``_judd_wright``
        / ``_power_law`` / ``_linear`` bound method (and therefore go through
        the existing layup-scaled coefficient table). A user-supplied
        callable is validated on a ``Vp ∈ [0, 1]`` grid (so the contract
        ``model(Vp, mode) -> float in [0, 1]`` is enforced once at dispatch
        time, not silently propagated to downstream nodal arrays) and
        returned as-is — the caller owns the coefficients, so the layup
        scaling is bypassed (#62).
        """
        if isinstance(model, str):
            _MODEL_FUNCS = {'judd_wright': self._judd_wright,
                            'power_law': self._power_law,
                            'linear': self._linear}
            if model not in _MODEL_FUNCS:
                raise ValueError(
                    f"Unknown knockdown model {model!r}. "
                    f"Use one of {sorted(_MODEL_FUNCS)} or pass a callable."
                )
            return _MODEL_FUNCS[model], False
        # User-supplied callable.
        self._validate_user_kd_callable(model, mode)
        return model, True

    def apply_loading(self, mode: str = 'compression',
                      model: Union[str, Callable[[float, str], float]] = 'judd_wright',
                      *,
                      cycles: Optional[float] = None,
                      environment: Optional[Dict[str, float]] = None,
                      R: Optional[float] = None):
        """Compute per-node knockdown for a given loading mode and model.

        Populates ``self.nodal_knockdown`` (shape ``(n_nodes,)``, values in
        ``(0, 1]``) by evaluating the empirical model at each node's local
        ``Vp`` and folding in any discrete-void stress concentration factors.
        When ``environment`` and / or ``cycles`` are supplied, the hygrothermal
        and / or S-N fatigue knockdowns are composed multiplicatively into
        the per-node field (issue #59).

        Parameters
        ----------
        mode : {'compression', 'tension', 'shear', 'ilss', 'transverse_tension'}
            Loading mode that selects the pristine strength and the
            mode-specific empirical coefficient.
        model : str or callable
            Empirical knockdown form. A string in
            ``{'judd_wright', 'power_law', 'linear'}`` dispatches to the
            built-in model with its layup-scaled coefficient. Alternatively,
            a callable matching the contract
            ``model(Vp: float, mode: str) -> float ∈ [0, 1]`` plugs in a
            user-defined knockdown law. User callables own their own
            coefficients, so the internal layup scaling is bypassed; the
            discrete-void SCF post-step still applies so caller-defined
            knockdowns and explicit voids compose the same way as the
            built-in models.
        cycles : int or float, optional
            Number of load cycles ``N`` for an S-N fatigue knockdown
            (issue #59). When ``None`` (default) no fatigue effect is
            applied. When supplied, the :class:`FatigueModel` log-linear
            knockdown is composed multiplicatively. Requires ``N >= 1``.
        environment : dict, optional
            Mapping with optional keys ``'T'`` (service temperature, deg C)
            and ``'M'`` (moisture content, wt%) for the hygrothermal
            knockdown (issue #59). When ``None`` (default) no environment
            effect is applied. The hygrothermal factor comes from
            :meth:`MaterialProperties.environment_knockdown`.
        R : float, optional
            Stress ratio for the fatigue knockdown. Currently
            informational; default (``None``) falls back to ``R = 0.1``
            (tension-tension).

        Notes
        -----
        Sign convention: ``mode='compression'`` and ``mode='ilss'`` return a
        **positive magnitude** failure stress (and a knockdown in ``(0, 1]``),
        not a signed value. Tension and compression strengths are stored on
        :class:`MaterialProperties` as positive numbers (``sigma_1c``,
        ``sigma_1t``), and the downstream
        ``failure_stress_MPa = KD * sigma_0`` is reported with the same
        positive-magnitude sign. Distinguish modes by the ``mode`` field of
        the returned dict, not by the sign of the stress. This is the
        empirical-solver convention only — :class:`FieldResults` stores
        **signed** FE stresses and strains in Voigt order
        ``[11, 22, 33, 23, 13, 12]`` (engineering shear).
        """
        if mode not in self.PRISTINE_STRENGTH_KEY:
            raise ValueError(
                f"Unknown loading mode {mode!r}. "
                f"Use one of {sorted(self.PRISTINE_STRENGTH_KEY)}."
            )
        # #115: vectorize the built-in knockdown evaluation. The scalar list
        # comprehension was ~60x slower than NumPy on the per-node Vp array
        # (4400-element typical mesh). User-supplied callables still get the
        # scalar path so the (Vp, mode) -> float contract from #62 holds.
        Vp_arr = self.mesh.porosity
        # Mesh.porosity is None before `generate_mesh()` runs; the solver
        # never gets that far without it, but narrow the type explicitly so
        # static checkers can see the vectorized branches are well-typed.
        assert Vp_arr is not None, "CompositeMesh.porosity not populated"
        if isinstance(model, str):
            # Mode validation already done above; this picks built-ins or
            # raises a clear error before we touch the array.
            if model == 'judd_wright':
                kd = np.exp(-self.JUDD_WRIGHT_ALPHA[mode] * Vp_arr)
            elif model == 'power_law':
                kd = (1.0 - Vp_arr) ** self.POWER_LAW_N[mode]
            elif model == 'linear':
                kd = np.maximum(1.0 - self.LINEAR_BETA[mode] * Vp_arr, 0.0)
            else:
                raise ValueError(
                    f"Unknown knockdown model {model!r}. "
                    f"Use one of ['judd_wright', 'power_law', 'linear'] "
                    f"or pass a callable."
                )
        else:
            self._validate_user_kd_callable(model, mode)
            kd = np.array([model(Vp, mode) for Vp in Vp_arr])
        kd = self._apply_discrete_void_scf(kd, mode)
        env_kd = self._environment_knockdown_factor(mode, environment)
        fat_kd = self._fatigue_knockdown_factor(mode, cycles, R)
        if env_kd != 1.0:
            kd = kd * env_kd
        if fat_kd != 1.0:
            kd = kd * fat_kd
        self.nodal_knockdown = kd  # type: ignore[assignment]  # lazy-init attr starts None

    def get_failure_load(self, mode: str = 'compression',
                         model: Union[str, Callable[[float, str], float]]
                         = 'judd_wright',
                         *,
                         cycles: Optional[float] = None,
                         environment: Optional[Dict[str, float]] = None,
                         R: Optional[float] = None) -> 'FailureResult':
        """Compute failure load using specimen-average porosity.

        The knockdown is evaluated at the mean Vp (matching how the original
        correlations were calibrated), not at the local peak.  Per-node
        knockdown is still computed for visualization via apply_loading().
        Optional hygrothermal (``environment``) and S-N fatigue
        (``cycles`` / ``R``) knockdowns compose multiplicatively with the
        porosity knockdown (issue #59).

        Parameters
        ----------
        mode : str
            Loading mode; see :meth:`apply_loading`.
        model : str or callable
            Either a built-in name (``'judd_wright'``, ``'power_law'``,
            ``'linear'``) or a user-supplied callable with signature
            ``model(Vp: float, mode: str) -> float ∈ [0, 1]``. User
            callables bypass the layup-scaled coefficient table (the caller
            owns their model).
        cycles : int or float, optional
            Number of load cycles ``N`` for an S-N fatigue knockdown.
            When ``None`` (default) returns ``1.0`` for the fatigue
            factor (no fatigue effect). When supplied, the
            :class:`FatigueModel` log-linear knockdown composes
            multiplicatively with the porosity knockdown. The result
            ``details`` dict gains a ``'fatigue_knockdown'`` entry so the
            breakdown is auditable.
        environment : dict, optional
            Hygrothermal conditioning dict with optional keys ``'T'``
            (service temperature, deg C) and ``'M'`` (moisture content,
            wt%). When ``None`` (default) no hygrothermal knockdown is
            applied. Composed multiplicatively with porosity and fatigue;
            the result ``details`` dict gains an ``'environment_knockdown'``
            entry when active.
        R : float, optional
            Stress ratio for the fatigue knockdown. Currently informational
            (default ``None`` -> ``R = 0.1``).

        Returns
        -------
        FailureResult
            Unified failure summary (#44 item 1) with ``failure_stress``,
            ``knockdown``, ``model`` attributes plus a ``details`` dict
            carrying the legacy ``'critical_location'`` extra and the new
            ``'environment_knockdown'`` / ``'fatigue_knockdown'`` entries
            (when active). Back-compat dict-style access
            (``result['failure_stress']``, ``result['critical_location']``,
            etc.) is preserved via the :class:`FailureResult`
            ``__getitem__`` shim and will be removed in a future major
            version — prefer attribute access.
        """
        self.apply_loading(mode, model,
                           cycles=cycles, environment=environment, R=R)
        sigma_0 = self._get_pristine_strength(mode)

        # Use specimen-average Vp for knockdown (matches calibration basis)
        Vp_mean = self.mesh.porosity_field.Vp
        model_func, _is_user = self._resolve_knockdown_model(model, mode)
        porosity_kd = float(model_func(Vp_mean, mode))

        # Hygrothermal and fatigue knockdowns compose multiplicatively at
        # the same point as the porosity knockdown so the final
        # `failure_stress` carries all three effects, mirroring how the
        # layup scaling is folded into the empirical coefficients upstream.
        env_kd = self._environment_knockdown_factor(mode, environment)
        fat_kd = self._fatigue_knockdown_factor(mode, cycles, R)
        mean_kd = porosity_kd * env_kd * fat_kd

        # Record a JSON-friendly label even when the caller passes a
        # callable (lambdas/closures don't round-trip through json.dumps).
        model_label = model if isinstance(model, str) else getattr(
            model, '__name__', 'user_callable')

        details: Dict[str, Any] = {
            'critical_location': [0.0, 0.0, 0.0],
            'mode': mode,
        }
        # Surface the per-knockdown breakdown in ``details`` so callers
        # (e.g. the #65 tornado sensitivities) can audit the composition.
        if environment is not None:
            details['environment_knockdown'] = float(env_kd)
        if cycles is not None:
            details['fatigue_knockdown'] = float(fat_kd)

        return FailureResult(
            failure_stress=float(sigma_0 * mean_kd),
            knockdown=float(mean_kd),
            model=str(model_label),
            details=details,
        )

    def get_all_failure_loads(
            self,
            extra_models: Optional[Dict[str, Callable[[float, str], float]]] = None,
            *,
            cycles: Optional[float] = None,
            environment: Optional[Dict[str, float]] = None,
            R: Optional[float] = None,
    ) -> dict:
        """Compute failure loads for all modes against all built-in models.

        Parameters
        ----------
        extra_models : dict[str, callable], optional
            User-supplied knockdown callables to evaluate alongside the
            built-ins. Each entry's key is the label used in the result
            dict; its value must be a callable matching the contract
            ``model(Vp: float, mode: str) -> float ∈ [0, 1]``. The built-in
            three models are always included.
        cycles : int or float, optional
            Number of load cycles ``N`` for an S-N fatigue knockdown
            (issue #59). Threaded into every :meth:`get_failure_load` call.
            ``None`` -> no fatigue effect.
        environment : dict, optional
            Hygrothermal conditioning dict (see :meth:`apply_loading`).
            Threaded into every :meth:`get_failure_load` call. ``None`` ->
            no hygrothermal effect.
        R : float, optional
            Stress ratio for the fatigue knockdown (informational).
        """
        results: Dict[str, Dict[str, FailureResult]] = {}
        all_models: List[Tuple[str, Union[str, Callable[[float, str], float]]]] = [
            ('judd_wright', 'judd_wright'),
            ('power_law', 'power_law'),
            ('linear', 'linear'),
        ]
        if extra_models:
            for label, fn in extra_models.items():
                all_models.append((str(label), fn))
        for mode in ['compression', 'tension', 'shear', 'ilss',
                     'transverse_tension']:
            results[mode] = {}
            for label, model in all_models:
                results[mode][label] = self.get_failure_load(
                    mode, model,
                    cycles=cycles, environment=environment, R=R,
                )
        return results

    def local_sensitivities(self, mode: str = 'compression',
                            model: str = 'judd_wright',
                            Vp: Optional[float] = None) -> Dict[str, float]:
        """Closed-form local sensitivities of the empirical knockdown.

        Returns the analytic partials ``dKD/dVp`` and ``dKD/dcoef`` (the
        layup-scaled coefficient for the chosen model) at the supplied
        porosity ``Vp``.  The three knockdown laws are closed form, so the
        partials are exact, dimensionless, and effectively free to compute
        (no FD, no sampling).

        Partial-derivative table (with ``c`` denoting the layup-scaled
        coefficient — ``alpha`` for Judd-Wright, ``n`` for power-law,
        ``beta`` for linear):

        =============  ===========================  ===========================
        Model          ``dKD/dVp``                  ``dKD/dcoef``
        =============  ===========================  ===========================
        judd_wright    ``-alpha * KD``              ``-Vp * KD``
        power_law      ``-n * (1 - Vp)**(n-1)``     ``(1-Vp)**n * ln(1-Vp)``
        linear         ``-beta`` (or 0 if clipped)  ``-Vp``  (or 0 if clipped)
        =============  ===========================  ===========================

        Parameters
        ----------
        mode:
            Loading mode (same keys as :meth:`get_failure_load`).
        model:
            Empirical knockdown law: ``'judd_wright'``, ``'power_law'``,
            or ``'linear'``.
        Vp:
            Specimen-average porosity at which to evaluate.  Defaults to
            ``self.mesh.porosity_field.Vp`` — the same value
            :meth:`get_failure_load` uses.

        Returns
        -------
        dict
            ``{'KD': float, 'dKD_dVp': float, 'dKD_dcoef': float}``.
            ``dKD_dcoef`` is the partial with respect to the layup-scaled
            coefficient (alpha/n/beta) that the solver actually applied;
            it already reflects the layup scaling from
            :meth:`_layup_scale`.
        """
        if mode not in self.PRISTINE_STRENGTH_KEY:
            raise ValueError(
                f"Unknown loading mode {mode!r}. "
                f"Use one of {sorted(self.PRISTINE_STRENGTH_KEY)}."
            )
        if Vp is None:
            Vp = self.mesh.porosity_field.Vp
        Vp = self._check_internal_Vp(Vp)
        if model == 'judd_wright':
            alpha = self.JUDD_WRIGHT_ALPHA[mode]
            kd = float(np.exp(-alpha * Vp))
            return {
                'KD': kd,
                'dKD_dVp': float(-alpha * kd),
                'dKD_dcoef': float(-Vp * kd),
            }
        if model == 'power_law':
            n = self.POWER_LAW_N[mode]
            one_minus = 1.0 - Vp
            kd = float(one_minus**n)
            # Guard the log when Vp = 1 (degenerate edge): KD is 0 there
            # and the d/dn partial collapses to 0 because KD * ln(1-Vp)
            # is 0 * (-inf) in the limit.  We pin it to 0.0 explicitly so
            # callers see a finite value.
            if one_minus <= 0.0:
                d_dcoef = 0.0
                d_dVp = 0.0
            else:
                d_dcoef = float(kd * np.log(one_minus))
                d_dVp = float(-n * one_minus**(n - 1.0))
            return {
                'KD': kd,
                'dKD_dVp': d_dVp,
                'dKD_dcoef': d_dcoef,
            }
        if model == 'linear':
            beta = self.LINEAR_BETA[mode]
            raw = 1.0 - beta * Vp
            kd = float(max(raw, 0.0))
            # The linear law is clipped at 0: once raw < 0, the
            # piecewise-constant 0 floor has zero gradient.
            if raw <= 0.0:
                d_dVp = 0.0
                d_dcoef = 0.0
            else:
                d_dVp = float(-beta)
                d_dcoef = float(-Vp)
            return {
                'KD': kd,
                'dKD_dVp': d_dVp,
                'dKD_dcoef': d_dcoef,
            }
        raise ValueError(
            f"Unknown knockdown model {model!r}. "
            f"Use one of ['judd_wright', 'linear', 'power_law']."
        )

    def sensitivity_fd(self, mode: str = 'compression',
                       model: str = 'judd_wright',
                       param: str = 'Vp',
                       h: float = 1e-4) -> float:
        """Central-difference fallback for the local knockdown sensitivity.

        Useful for paths where the gradient is not closed form (FE or
        Mori-Tanaka couplings).  For the bundled empirical models this
        matches :meth:`local_sensitivities` to ~1e-7 and is shipped as a
        cross-check / drop-in for non-analytic models.

        Parameters
        ----------
        mode:
            Loading mode (same keys as :meth:`get_failure_load`).
        model:
            Empirical knockdown law.
        param:
            Either ``'Vp'`` (porosity) or ``'coef'`` (the layup-scaled
            coefficient that the model actually applied — alpha/n/beta).
        h:
            Step size for the central difference.

        Returns
        -------
        float
            ``(KD(x+h) - KD(x-h)) / (2*h)`` evaluated at the same
            ``Vp_mean`` :meth:`get_failure_load` uses.
        """
        if mode not in self.PRISTINE_STRENGTH_KEY:
            raise ValueError(
                f"Unknown loading mode {mode!r}. "
                f"Use one of {sorted(self.PRISTINE_STRENGTH_KEY)}."
            )
        if model not in ('judd_wright', 'power_law', 'linear'):
            raise ValueError(
                f"Unknown knockdown model {model!r}. "
                f"Use one of ['judd_wright', 'linear', 'power_law']."
            )
        if param not in ('Vp', 'coef'):
            raise ValueError(
                f"param must be 'Vp' or 'coef', got {param!r}."
            )
        Vp0 = float(self.mesh.porosity_field.Vp)

        # Select the analytic functional form so we can perturb the
        # parameter without mutating solver state.
        if model == 'judd_wright':
            coef0 = float(self.JUDD_WRIGHT_ALPHA[mode])
            def f(Vp_val, coef_val):
                return float(np.exp(-coef_val * Vp_val))
        elif model == 'power_law':
            coef0 = float(self.POWER_LAW_N[mode])
            def f(Vp_val, coef_val):
                return float((1.0 - Vp_val)**coef_val)
        else:  # linear
            coef0 = float(self.LINEAR_BETA[mode])
            def f(Vp_val, coef_val):
                return float(max(1.0 - coef_val * Vp_val, 0.0))

        if param == 'Vp':
            return float((f(Vp0 + h, coef0) - f(Vp0 - h, coef0)) / (2.0 * h))
        # param == 'coef'
        return float((f(Vp0, coef0 + h) - f(Vp0, coef0 - h)) / (2.0 * h))

