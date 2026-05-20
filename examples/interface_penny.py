"""Interface-concentrated porosity with penny-shaped voids.

Demonstrates the worst-case morphology for interlaminar shear strength
(ILSS): porosity peaked at every ply-to-ply interface (``distribution =
'interface'``) and shaped as oblate ``penny`` voids (high aspect ratio,
elevated stress-concentration factor). Penny voids amplify the through-
thickness stress concentration much more than spherical voids of the same
``Vp``. Saves the through-thickness porosity profile (a comb of peaks at
each interface) as a PNG.

Run from the repo root::

    python examples/interface_penny.py
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
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    material = MATERIALS["T800_epoxy"]
    pf = PorosityField(material, void_volume_fraction=0.03,
                       distribution="interface",
                       void_shape="penny")
    mesh = CompositeMesh(pf, material, nx=30, ny=10, nz=24)
    solver = EmpiricalSolver(mesh, material)
    res_ilss = solver.get_failure_load(mode="ilss", model="judd_wright")
    res_comp = solver.get_failure_load(mode="compression", model="judd_wright")

    z, Vp_z = pf.effective_porosity_profile(nz=400)
    print("Material:           T800_epoxy")
    print(f"Vp (mean):          {pf.Vp * 100:.2f}%   (interface, penny)")
    print(f"Vp (peak):          {Vp_z.max() * 100:.2f}%  at z = "
          f"{z[Vp_z.argmax()]:.3f} mm")
    print(f"ILSS knockdown:     {res_ilss['knockdown']:.4f}")
    print(f"ILSS strength:      {res_ilss['failure_stress']:.1f} MPa  "
          f"(positive magnitude)")
    print(f"Compression KD:     {res_comp['knockdown']:.4f}")
    print(f"Compression sigma:  {res_comp['failure_stress']:.1f} MPa")

    out_png = os.path.join(OUT_DIR, "interface_penny_profile.png")
    FEVisualizer.plot_porosity_field(pf, save_path=out_png)
    print(f"PNG saved:          {out_png}")


if __name__ == "__main__":
    main()
