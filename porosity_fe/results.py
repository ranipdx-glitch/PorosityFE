"""Shared result dataclasses (FailureResult / ConfigResult / ConfigArtifacts).

Lives in its own module so :mod:`porosity_fe.empirical`,
:mod:`porosity_fe.fe.solver` and :mod:`porosity_fe.pipeline` can all
consume them without importing each other (#119).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .empirical import EmpiricalSolver
    from .fe.solver import FieldResults
    from .mesh import CompositeMesh
    from .porosity_field import PorosityField

@dataclass
class FailureResult:
    """Unified failure-load summary returned by empirical and FE solvers.

    Adds API consistency across :meth:`EmpiricalSolver.get_failure_load`
    (historically dict-returning) and :meth:`FESolver.solve` (returns the
    richer :class:`FieldResults`, distilled here via
    :meth:`FieldResults.summary`). Callers can now treat the two solvers
    polymorphically (#44 item 1).

    Attributes
    ----------
    failure_stress : float
        Failure stress magnitude in MPa. For the empirical solver this is
        ``knockdown * sigma_pristine`` at the specimen-average ``Vp``; for
        the FE solver (distilled via :meth:`FieldResults.summary`) it is
        ``knockdown * sigma_pristine`` evaluated with the loading-mode-
        specific pristine strength. Reported as a positive magnitude
        regardless of loading sign (compression strengths are stored as
        positive numbers in :class:`MaterialProperties`).
    knockdown : float
        Knockdown factor in ``(0, 1]``. Same definition the source solver
        used; bit-identical to what the legacy dict / float returns
        produced.
    model : str
        Knockdown model label (``'judd_wright'`` / ``'power_law'`` /
        ``'linear'`` / ``'user_callable'`` for the empirical solver, and
        ``'fe_<criterion>'`` for the FE solver's summary).
    details : dict
        Free-form solver-specific extras (e.g. ``'critical_location'`` for
        the empirical path, or ``'max_failure_index'`` /
        ``'failure_criterion'`` from the FE solver). Always JSON-friendly.

    Notes
    -----
    Back-compat: callers that historically accessed
    ``result['failure_stress']`` / ``result['knockdown']`` / ``result['model']``
    / ``result['critical_location']`` keep working via the ``__getitem__``
    shim below. The shim maps the four documented dict keys to attribute /
    ``details`` access; any other key raises :class:`KeyError`. New code
    should use attribute access. The dict shim will be removed in a future
    major version.
    """
    failure_stress: float
    knockdown: float
    model: str
    details: dict = field(default_factory=dict)

    # Dict keys served by the back-compat shim. ``critical_location`` is
    # routed through ``details`` because not every distilled FailureResult
    # has a meaningful crack location (the FE summary uses a max-FI
    # element index instead).
    _DICT_KEYS_DIRECT = ('failure_stress', 'knockdown', 'model')

    def __getitem__(self, key: str):
        """Back-compat dict-style access (deprecated; will be removed)."""
        if key in self._DICT_KEYS_DIRECT:
            return getattr(self, key)
        if key in self.details:
            return self.details[key]
        raise KeyError(
            f"{key!r} is not a known FailureResult field. "
            f"Known: {sorted(set(self._DICT_KEYS_DIRECT) | set(self.details))}."
        )

    def __contains__(self, key: object) -> bool:
        return key in self._DICT_KEYS_DIRECT or key in self.details

    def get(self, key: str, default=None):
        """Dict-style :meth:`dict.get` for the back-compat shim."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Iterable of known dict-style keys (back-compat shim)."""
        return list(self._DICT_KEYS_DIRECT) + list(self.details.keys())

    def to_dict(self) -> dict:
        """Return a plain dict matching the legacy
        :meth:`EmpiricalSolver.get_failure_load` return shape."""
        out = {k: getattr(self, k) for k in self._DICT_KEYS_DIRECT}
        out.update(self.details)
        return out


@dataclass
class ConfigArtifacts:
    """Live solver objects retained for one configuration in a sweep.

    Returned in the second slot of the :func:`compare_configurations` tuple
    when ``return_artifacts=True``. Keeping these out of the default
    :class:`ConfigResult` lets batch loops over many sweeps hold only the
    headline numbers in memory (#44 item 3).

    Attributes
    ----------
    mesh : CompositeMesh
        The mesh used for the empirical / FE solves.
    empirical_solver : EmpiricalSolver
        The empirical solver constructed for this configuration.
    porosity_field : PorosityField
        The porosity field used for this configuration.
    field_results : FieldResults or None
        Populated only if the FE solver was run for this configuration
        (``compare_configurations`` is empirical-only today, so this is
        ``None`` from that path).
    """
    mesh: 'CompositeMesh'
    empirical_solver: 'EmpiricalSolver'
    porosity_field: 'PorosityField'
    field_results: Optional['FieldResults'] = None


@dataclass
class ConfigResult:
    """Lightweight (numbers-only) per-configuration result from
    :func:`compare_configurations`.

    Holds only JSON-friendly scalars plus the nested ``empirical`` dict
    (the headline knockdown / failure_stress tables already produced by
    :meth:`EmpiricalSolver.get_all_failure_loads`). Heavy live objects
    (mesh, empirical_solver, porosity_field) are kept on the parallel
    :class:`ConfigArtifacts` mapping, returned only when
    ``compare_configurations(..., return_artifacts=True)`` is requested.

    Attributes
    ----------
    Vp : float
        Specimen-average void volume fraction in [0, 1].
    config_name : str
        Configuration name (key in :data:`POROSITY_CONFIGS`).
    config : dict
        :class:`PorosityField` constructor kwargs for this configuration.
    failure_stress : float
        Headline compression failure stress (Judd-Wright model) in MPa.
        Convenience scalar; the full per-mode/per-model table is on
        :attr:`empirical`.
    knockdown : float
        Headline compression knockdown (Judd-Wright). Convenience scalar.
    model : str
        Label of the headline knockdown model. Always ``'judd_wright'``
        for the default sweep; preserved as a field so future overrides
        flow through.
    empirical : dict
        The nested empirical-knockdown table from
        :meth:`EmpiricalSolver.get_all_failure_loads`; structure is
        ``{mode: {model: FailureResult}}``. Kept on the result so the
        existing JSON exporter, plot helpers, and tests can continue to
        read ``cfg['empirical']['compression']['judd_wright']['knockdown']``
        without touching the artifacts dict.
    seed : int or None
        The reproducibility seed recorded on the underlying
        :class:`PorosityField` (mirrors the input to
        :func:`compare_configurations`). Carried on the lightweight
        result so the JSON exporter's provenance block can recover it
        without holding the live ``porosity_field`` (#55 / #44 item 3).

    Notes
    -----
    Back-compat: this object supports dict-style item access for the
    documented keys above (``'Vp'``, ``'config'``, ``'config_name'``,
    ``'failure_stress'``, ``'knockdown'``, ``'model'``, ``'empirical'``,
    ``'seed'``) so legacy callers do not break. Any *other* key —
    notably the legacy ``'mesh'`` / ``'empirical_solver'`` /
    ``'porosity_field'`` keys — raises :class:`KeyError` with a hint
    pointing at ``return_artifacts=True``. The dict shim will be removed
    in a future major version.
    """
    Vp: float
    config_name: str
    config: dict
    failure_stress: float
    knockdown: float
    model: str
    empirical: dict
    seed: Optional[int] = None

    _DICT_KEYS_DIRECT = (
        'Vp', 'config_name', 'config', 'failure_stress',
        'knockdown', 'model', 'empirical', 'seed',
    )
    # Old keys callers used to find via the dict — surface a helpful
    # KeyError now that they live on :class:`ConfigArtifacts`.
    _ARTIFACT_KEYS = ('mesh', 'empirical_solver', 'porosity_field', 'field_results')

    def __getitem__(self, key: str):
        if key in self._DICT_KEYS_DIRECT:
            return getattr(self, key)
        if key in self._ARTIFACT_KEYS:
            raise KeyError(
                f"{key!r} is no longer carried on the default "
                f"compare_configurations result (#44). Re-run with "
                f"`return_artifacts=True` and read it from the parallel "
                f"artifacts dict."
            )
        raise KeyError(
            f"{key!r} is not a known ConfigResult field. "
            f"Known: {sorted(self._DICT_KEYS_DIRECT)}."
        )

    def __contains__(self, key: object) -> bool:
        return key in self._DICT_KEYS_DIRECT

    def get(self, key: str, default=None):
        """Dict-style :meth:`dict.get` for the back-compat shim."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Iterable of known dict-style keys (back-compat shim)."""
        return list(self._DICT_KEYS_DIRECT)

