"""Shared ``_resolve_ply_angles`` helper and canonical baselines.

Extracted into its own module so :mod:`porosity_fe.mesh`,
:mod:`porosity_fe.empirical` and :mod:`porosity_fe.fe.solver` can all
consume the same sentinel-resolution logic without an import cycle.
"""

import warnings
from typing import List, Optional, Tuple, Union

#: Canonical QI baseline layup (8-ply symmetric ``[0/90/45/-45]_s``).
#: Used to expand the ``ply_angles='QI'`` sentinel (#44 item 2).
_PLY_ANGLES_QI: Tuple[float, ...] = (0.0, 90.0, 45.0, -45.0, -45.0, 45.0, 90.0, 0.0)

#: Canonical UD baseline (4 plies, all 0 deg). Used to expand the
#: ``ply_angles='UD'`` sentinel; the FE / empirical scaling only cares about
#: the angle distribution, so a short list is fine.
_PLY_ANGLES_UD: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


def _resolve_ply_angles(
    ply_angles: Optional[Union[List[float], Tuple[float, ...], str]],
    *,
    none_means: str = 'QI',
    caller: str = 'ply_angles',
) -> Optional[List[float]]:
    """Resolve the ``ply_angles`` sentinel to a concrete list of angles.

    Unifies the three previously divergent ``ply_angles=None`` defaults
    across :class:`EmpiricalSolver`, :class:`CompositeMesh` and
    :class:`FESolver` (#44 item 2). The resolved value is:

    - ``'QI'`` -> ``[0, 90, 45, -45, -45, 45, 90, 0]`` (8-ply symmetric
      quasi-isotropic). The standardised default for new code.
    - ``'UD'`` -> ``[0, 0, 0, 0]`` (unidirectional).
    - A list / tuple of floats -> returned verbatim (as a list).
    - ``None`` -> resolved to ``none_means`` (default ``'QI'``) and emits a
      :class:`DeprecationWarning`. Class-specific call sites override
      ``none_means`` if they need to preserve the prior class-specific
      default during the deprecation window.

    Returns
    -------
    list of float or None
        ``None`` is returned for :class:`CompositeMesh`'s historical
        ``None`` -> all-zero behaviour when ``none_means='UD_legacy'``; the
        empirical / FE paths always get an explicit list back.
    """
    if isinstance(ply_angles, str):
        key = ply_angles.upper()
        if key == 'QI':
            return list(_PLY_ANGLES_QI)
        if key == 'UD':
            return list(_PLY_ANGLES_UD)
        raise ValueError(
            f"{caller} string sentinel must be 'QI' or 'UD', got {ply_angles!r}."
        )
    if ply_angles is None:
        # Back-compat shim — emit a DeprecationWarning and resolve to the
        # standardised default. Planned removal in a future major version.
        warnings.warn(
            f"Passing {caller}=None is deprecated; pass {none_means!r} (or "
            f"'UD', or an explicit list of ply angles) instead. None is "
            f"resolved to {none_means!r} for back-compat and will be removed "
            "in a future major version (#44).",
            DeprecationWarning,
            stacklevel=3,
        )
        if none_means == 'QI':
            return list(_PLY_ANGLES_QI)
        if none_means == 'UD':
            return list(_PLY_ANGLES_UD)
        # 'UD_legacy' preserves the historical "None means literal zero array"
        # behaviour for CompositeMesh — same as 'UD' angle-wise.
        if none_means == 'UD_legacy':
            return None
        raise ValueError(
            f"Internal: unsupported none_means={none_means!r}."
        )
    # Concrete sequence — convert to list of floats for hashability /
    # reproducibility and validate entries.
    angle_list = [float(a) for a in ply_angles]
    return angle_list
