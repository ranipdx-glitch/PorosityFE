"""Uniform porosity with spherical voids — empirical knockdown screening.

Demonstrates the simplest configuration: a uniform 3% void volume fraction
of spherical voids in a T800 / epoxy laminate, evaluated under compression
with the calibrated Judd-Wright model. Prints the resulting knockdown and
failure stress (positive magnitude — see ``EmpiricalSolver.apply_loading``
Notes for the sign convention), and saves the through-thickness porosity
profile (flat for ``distribution='uniform'``) as a PNG.

Run from the repo root::

    python examples/uniform_spherical.py
"""

import os
import sys

# Make the sibling porosity_fe package importable when it is not
# pip-installed (matches the project's conftest.py path adjustment).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from porosity_fe import (  # noqa: E402
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FEVisualizer,
    PorosityField,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    material = MATERIALS["T800_epoxy"]
    pf = PorosityField(material, void_volume_fraction=0.03,
                       distribution="uniform", void_shape="spherical")
    mesh = CompositeMesh(pf, material, nx=30, ny=10, nz=12)
    solver = EmpiricalSolver(mesh, material)
    result = solver.get_failure_load(mode="compression", model="judd_wright")

    sigma_0 = material.sigma_1c
    print(f"Material:        T800_epoxy   (sigma_1c = {sigma_0:.1f} MPa)")
    print(f"Vp:              {pf.Vp * 100:.2f}%   (uniform, spherical)")
    print(f"Knockdown:       {result['knockdown']:.4f}")
    print(f"Failure stress:  {result['failure_stress']:.1f} MPa  "
          f"(positive magnitude, compression mode)")

    out_png = os.path.join(OUT_DIR, "uniform_spherical_profile.png")
    FEVisualizer.plot_porosity_field(pf, save_path=out_png)
    print(f"PNG saved:       {out_png}")


if __name__ == "__main__":
    main()
