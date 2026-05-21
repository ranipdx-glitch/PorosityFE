"""CLT path: laminate effective moduli ``(Ex, Ey, Gxy)`` vs porosity.

Demonstrates the Classical Lamination Theory degradation path via
:func:`compute_degraded_clt_moduli`. The function applies Mori-Tanaka
stiffness degradation to a single ply (matrices in the Voigt order
``[11, 22, 33, 23, 13, 12]`` with engineering shear convention — see the
Notes block on :meth:`MaterialProperties.get_stiffness_matrix`), then
builds the laminate A-matrix and extracts the in-plane effective moduli
``Ex``, ``Ey``, ``Gxy``. Sweeps ``Vp`` over ``[0, 5%]`` and saves a plot
of the three moduli vs. porosity, all normalized by their pristine values.

Run from the repo root::

    python examples/compute_degraded_clt_moduli.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from porosity_fe import MATERIALS, compute_degraded_clt_moduli  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    material = MATERIALS["T800_epoxy"]
    ply_angles = [0.0, 45.0, -45.0, 90.0,
                  90.0, -45.0, 45.0, 0.0]  # [0/45/-45/90]_s
    Vps = np.linspace(0.0, 0.05, 11)
    Ex, Ey, Gxy = [], [], []
    for Vp in Vps:
        m = compute_degraded_clt_moduli(material, ply_angles, float(Vp))
        Ex.append(m["Ex"])
        Ey.append(m["Ey"])
        Gxy.append(m["Gxy"])
    Ex = np.array(Ex)
    Ey = np.array(Ey)
    Gxy = np.array(Gxy)

    print("Layup: [0/45/-45/90]_s   (8 plies, T800_epoxy)")
    print(f"{'Vp(%)':>7s} {'Ex/Ex0':>9s} {'Ey/Ey0':>9s} {'Gxy/Gxy0':>11s}")
    for v, ex, ey, g in zip(Vps, Ex / Ex[0], Ey / Ey[0], Gxy / Gxy[0]):
        print(f"{v*100:7.2f} {ex:9.4f} {ey:9.4f} {g:11.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Vps * 100, Ex / Ex[0], "o-", label="Ex / Ex0")
    ax.plot(Vps * 100, Ey / Ey[0], "s-", label="Ey / Ey0")
    ax.plot(Vps * 100, Gxy / Gxy[0], "^-", label="Gxy / Gxy0")
    ax.set_xlabel("Void volume fraction Vp (%)")
    ax.set_ylabel("Normalized modulus")
    ax.set_title("CLT-degraded laminate moduli vs. porosity")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.05)
    out_png = os.path.join(OUT_DIR, "clt_degraded_moduli.png")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"PNG saved: {out_png}")


if __name__ == "__main__":
    main()
