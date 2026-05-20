"""S-N (fatigue) knockdown model (#59)."""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# ============================================================
# Fatigue (S-N) knockdown surface (issue #59).
#
# Log-linear (Mandell) fatigue knockdown::
#
#     S_N / S_0 = max(floor, 1 - b * log10(N))
#
# with mode-keyed slopes ``b``. The table below is calibrated against
# representative CFRP S-N data at ``R = 0.1`` (tension-tension): the
# tension/compression slopes follow the Mandell 1991 review value of
# b ~ 0.1 per decade for IM-class CFRP; ILSS / matrix-dominated shear
# is slightly shallower (b ~ 0.08) as reported by Curtis (1989) and the
# WWFE-III fatigue exercise. These are screening-level values; production
# allowables should use a fully populated S-N matrix per CMH-17 Vol. 2.
#
# References:
# - Mandell, J. F., "Fatigue Behavior of Fiber-Resin Composites,"
#   Developments in Reinforced Plastics 2, 1991.
# - Curtis, P. T., "The fatigue behaviour of fibrous composite materials,"
#   J. Strain Analysis, 1989.
# ============================================================
_FATIGUE_B_QI: Dict[str, float] = {
    'tension': 0.10,
    'compression': 0.10,
    'shear': 0.08,
    'ilss': 0.08,
    'transverse_tension': 0.10,
}

# Floor clamp for the log-linear formula: at very large N the linear
# extrapolation predicts a negative knockdown. Clamp to 1% of static so
# downstream multiplications stay well-behaved, and emit a warning so the
# caller knows they are off the calibration range.
_FATIGUE_KD_FLOOR = 0.01


@dataclass
class FatigueModel:
    """S-N (cycles-to-failure) knockdown surface.

    Implements the log-linear (Mandell-style) form::

        S_N / S_0 = max(floor, 1 - b * log10(N))

    where ``b`` is a mode-keyed slope (see :data:`_FATIGUE_B_QI`),
    ``N`` is the number of load cycles, and ``floor`` (default 0.01)
    is a small lower clamp that prevents the linear extrapolation from
    going negative at very large ``N``.

    The default slopes are calibrated for quasi-isotropic CFRP at
    ``R = 0.1`` (typical tension-tension fatigue). The ``R`` argument is
    currently informational only — future revisions can wrap the base
    formula with a Goodman / Walker R-correction.

    Attributes
    ----------
    b : dict[str, float], optional
        Mode-keyed slope override. Modes absent from the override fall
        back to :data:`_FATIGUE_B_QI`.

    Notes
    -----
    For ``cycles = None`` callers should bypass this class entirely (the
    :meth:`EmpiricalSolver.get_failure_load` path returns ``1.0`` in that
    case). The model is screening-level; production allowables should
    come from a fully populated test matrix per the applicable spec
    (e.g. CMH-17 Vol. 2 fatigue protocols).
    """
    b: Optional[Dict[str, float]] = None

    def _slope(self, mode: str) -> float:
        if mode not in _FATIGUE_B_QI:
            raise ValueError(
                f"Unknown fatigue mode {mode!r}. "
                f"Use one of {sorted(_FATIGUE_B_QI)}."
            )
        if self.b is not None and mode in self.b:
            return float(self.b[mode])
        return float(_FATIGUE_B_QI[mode])

    def knockdown_factor(self, mode: str, cycles: float,
                         R: float = 0.1) -> float:
        """Multiplicative fatigue knockdown for the given mode.

        Parameters
        ----------
        mode : str
            Loading mode (see :data:`_FATIGUE_B_QI`).
        cycles : float
            Number of load cycles ``N``. Must be a positive finite
            value; ``cycles = 1`` corresponds to the static (one-cycle)
            allowable and returns ``1.0``.
        R : float, optional
            Stress ratio ``sigma_min / sigma_max``. Currently
            informational (default 0.1, tension-tension); reserved
            for a future Goodman / Walker R-correction.

        Returns
        -------
        float
            Knockdown factor in ``[floor, 1.0]``. When the linear
            extrapolation would go below the floor (``floor = 0.01``),
            the value is clamped and a :class:`UserWarning` is emitted.
        """
        # Reserved for future R-correction; today it is purely
        # informational. Validate finiteness so a stray nan can't slip
        # through silently.
        if not np.isfinite(float(R)):
            raise ValueError(
                f"FatigueModel.knockdown_factor: R must be finite, got {R!r}."
            )

        N = float(cycles)
        if not np.isfinite(N) or N < 1.0:
            raise ValueError(
                f"FatigueModel.knockdown_factor: cycles must be a finite "
                f"value >= 1, got {cycles!r}."
            )

        b = self._slope(mode)
        raw = 1.0 - b * np.log10(N)
        if raw < _FATIGUE_KD_FLOOR:
            warnings.warn(
                f"Fatigue knockdown for mode={mode!r}, cycles={N:.3g}, "
                f"R={R!r} extrapolates to {raw:.3g} (<= floor "
                f"{_FATIGUE_KD_FLOOR}); clamping. The log-linear model is "
                f"off its calibration range — consider a richer S-N model.",
                UserWarning,
                stacklevel=2,
            )
            return _FATIGUE_KD_FLOOR
        return float(min(raw, 1.0))
