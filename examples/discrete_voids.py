"""Explicit discrete voids (``VoidGeometry``) on top of a uniform field.

Demonstrates the discrete-void path: two explicit ellipsoidal voids are
placed in addition to a low (``Vp = 0.5%``) uniform background, and the
empirical solver applies the void-specific stress-concentration factors at
each node. Also renders the SCF field around one of the voids as the PNG
output (uses ``FEVisualizer.plot_void_scf``, which shows how the SCF
decays away from a single ellipsoid).

Run from the repo root::

    python examples/discrete_voids.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from porosity_fe_analysis import (  # noqa: E402
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FEVisualizer,
    PorosityField,
    VoidGeometry,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    material = MATERIALS["T800_epoxy"]
    # Two discrete ellipsoidal voids: a roughly spherical one near the
    # midplane and a more oblate (penny-like) one offset in y.
    voids = [
        VoidGeometry(center=(25.0, 10.0, 2.20),
                     radii=(1.5, 1.5, 1.2)),
        VoidGeometry(center=(35.0,  6.0, 2.20),
                     radii=(2.5, 2.0, 0.6)),
    ]
    pf = PorosityField(material, void_volume_fraction=0.005,
                       distribution="uniform", void_shape="spherical",
                       discrete_voids=voids)
    mesh = CompositeMesh(pf, material, nx=40, ny=15, nz=12)
    solver = EmpiricalSolver(mesh, material)
    result = solver.get_failure_load(mode="compression", model="judd_wright")

    print("Material:        T800_epoxy")
    print(f"Vp (background): {pf.Vp * 100:.2f}%   "
          f"({len(voids)} discrete voids on top)")
    print("Discrete voids:")
    for i, v in enumerate(voids):
        scf = v.stress_concentration_factor()["compression"]
        print(f"  void {i}: center={v.center.tolist()}, "
              f"radii={v.radii.tolist()}, AR={v.aspect_ratio:.2f}, "
              f"SCF_compression={scf:.2f}")
    print(f"Knockdown:       {result['knockdown']:.4f}")
    print(f"Failure stress:  {result['failure_stress']:.1f} MPa  "
          f"(positive magnitude, compression mode)")

    out_png = os.path.join(OUT_DIR, "discrete_voids_scf.png")
    FEVisualizer.plot_void_scf(voids[1], save_path=out_png)
    print(f"PNG saved:       {out_png}")


if __name__ == "__main__":
    main()
