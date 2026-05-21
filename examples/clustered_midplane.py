"""Gaussian-clustered porosity at the laminate midplane.

Demonstrates a heterogeneous through-thickness profile: the same specimen-
average ``Vp = 3%`` of spherical voids as ``uniform_spherical.py``, but
concentrated in a Gaussian peak at the midplane (``cluster_location =
'midplane'``). The profile is normalized so the through-thickness mean
still equals the input ``Vp`` (so the empirical knockdown is the same as
uniform — the clustering matters for the FE path, not the calibrated
empirical correlation). Saves the through-thickness porosity profile so
the clustered shape is visually distinct from the uniform case.

Run from the repo root::

    python examples/clustered_midplane.py
"""

import os
import sys

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
                       distribution="clustered",
                       void_shape="spherical",
                       cluster_location="midplane")
    mesh = CompositeMesh(pf, material, nx=30, ny=10, nz=24)
    solver = EmpiricalSolver(mesh, material)
    result = solver.get_failure_load(mode="ilss", model="judd_wright")

    z, Vp_z = pf.effective_porosity_profile(nz=100)
    print("Material:        T800_epoxy")
    print(f"Vp (mean):       {pf.Vp * 100:.2f}%   (clustered@midplane, spherical)")
    print(f"Vp (peak):       {Vp_z.max() * 100:.2f}%  at z = "
          f"{z[Vp_z.argmax()]:.2f} mm")
    print(f"ILSS knockdown:  {result['knockdown']:.4f}")
    print(f"ILSS strength:   {result['failure_stress']:.1f} MPa  "
          f"(positive magnitude)")

    out_png = os.path.join(OUT_DIR, "clustered_midplane_profile.png")
    FEVisualizer.plot_porosity_field(pf, save_path=out_png)
    print(f"PNG saved:       {out_png}")


if __name__ == "__main__":
    main()
