"""Compare ``uniform`` vs ``clustered`` / ``interface`` porosity distributions.

Quantitative companion to GitHub issue #83 ("Investigate/document the
difference between the 'uniform' and stacked/clustered porosity
distributions"). For a matched specimen-average ``Vp = 3%`` across the
four supported through-thickness distributions
(``uniform``, ``clustered (midplane)``, ``clustered (surface)``,
``interface`` with ``void_shape='penny'``), this script reports empirical
knockdowns for every (mode, model) combination plus an FE-recovered
knockdown for the two most porosity-sensitive modes (compression and
ILSS).

The key finding is that the empirical solver uses the *specimen-average*
``Vp`` for the knockdown evaluation (see
:meth:`EmpiricalSolver.get_failure_load` — it reads
``self.mesh.porosity_field.Vp`` directly), so all four distributions
collapse to identical empirical numbers at matched mean Vp. The FE
solver, by contrast, integrates the local stiffness reduction over the
mesh and so picks up the peak-vs-mean difference, producing distinct
knockdowns for ``uniform`` vs ``clustered (midplane)`` even when their
specimen-average ``Vp`` is identical. This script exists to make that
contrast quantitative and reproducible — and to disambiguate the
("stack") terminology raised in issue #83: there is no preset literally
named ``stack``; the stacked / layered shapes are ``clustered`` and
``interface``.

Run from the repo root::

    python examples/distribution_comparison.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from porosity_fe import (  # noqa: E402
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FESolver,
    PorosityField,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Matched specimen-average void volume fraction (3%).
VP_MEAN = 0.03

# Distribution configurations: (label, kwargs forwarded to PorosityField).
# Each preserves ``void_volume_fraction = VP_MEAN`` so the specimen-mean
# Vp matches across all four; only the through-thickness *shape* differs.
DISTRIBUTIONS = [
    ("uniform", dict(distribution="uniform", void_shape="spherical")),
    ("clustered (midplane)", dict(distribution="clustered",
                                  cluster_location="midplane",
                                  void_shape="spherical")),
    ("clustered (surface)", dict(distribution="clustered",
                                 cluster_location="surface",
                                 void_shape="spherical")),
    ("interface", dict(distribution="interface", void_shape="penny")),
]

MODES = ("compression", "tension", "shear", "ilss", "transverse_tension")
MODELS = ("judd_wright", "power_law", "linear")
FE_MODES = ("compression", "ilss")


def _build(material, kwargs, *, nx=12, ny=4, nz=12):
    """Build (PorosityField, CompositeMesh) at matched ``VP_MEAN``."""
    pf = PorosityField(material, VP_MEAN, **kwargs)
    mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=nz)
    return pf, mesh


def _empirical_table(material):
    """Empirical KD for every (distribution, mode, model) at VP_MEAN."""
    rows = []
    for label, kwargs in DISTRIBUTIONS:
        pf, mesh = _build(material, kwargs)
        solver = EmpiricalSolver(mesh, material)
        kds = solver.get_all_failure_loads()
        for mode in MODES:
            for model in MODELS:
                rows.append({
                    "distribution": label,
                    "mode": mode,
                    "model": model,
                    "knockdown": float(kds[mode][model].knockdown),
                })
    return rows


def _fe_table(material):
    """FE knockdown for the two porosity-sensitive modes per distribution.

    The FE knockdown comes from the field solve, so it picks up the local
    peak Vp — unlike the empirical path which collapses everything to the
    specimen-mean.
    """
    rows = []
    fields = {}  # label -> (mesh, FieldResults_for_compression)
    for label, kwargs in DISTRIBUTIONS:
        pf, mesh = _build(material, kwargs)
        solver = FESolver(mesh, material, pf, ply_angles="QI")
        for mode in FE_MODES:
            if mode == "compression":
                res = solver.solve(loading="compression",
                                    applied_strain=-0.001)
                if label not in fields:
                    fields[label] = (mesh, res)
            elif mode == "ilss":
                res = solver.solve(loading="ilss", applied_load=-10.0)
            rows.append({
                "distribution": label,
                "mode": mode,
                "model": "fe_tsai_wu",
                "knockdown": float(res.knockdown),
            })
    return rows, fields


def _print_table(rows, *, group_by_mode=True):
    """Pretty-print a (distribution x mode x model) -> knockdown table."""
    dist_labels = [d[0] for d in DISTRIBUTIONS]
    header_w = 22
    col_w = 22
    if group_by_mode:
        # One subtable per mode; rows = model; columns = distribution.
        modes_in = sorted({r["mode"] for r in rows},
                          key=lambda m: (m not in MODES, m))
        models_in = sorted({r["model"] for r in rows})
        for mode in modes_in:
            print(f"\n  mode = {mode}")
            print("  " + "model".ljust(header_w)
                  + "".join(d.ljust(col_w) for d in dist_labels))
            print("  " + "-" * (header_w + col_w * len(dist_labels)))
            for model in models_in:
                # Only print rows that exist for this mode.
                values = []
                for d in dist_labels:
                    match = [r for r in rows
                             if r["mode"] == mode
                             and r["model"] == model
                             and r["distribution"] == d]
                    if not match:
                        values.append("—")
                    else:
                        values.append(f"{match[0]['knockdown']:.6f}")
                # Skip a model row that has no values at all.
                if all(v == "—" for v in values):
                    continue
                print("  " + model.ljust(header_w)
                      + "".join(v.ljust(col_w) for v in values))


def _plot_profiles(material, save_path):
    """Plot the through-thickness Vp profile for each distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (label, kwargs), c in zip(DISTRIBUTIONS, colors):
        pf = PorosityField(material, VP_MEAN, **kwargs)
        z, Vp = pf.effective_porosity_profile(nz=400)
        ax.plot(Vp * 100.0, z, label=label, color=c, linewidth=2)
    ax.axvline(VP_MEAN * 100.0, color="black", linestyle=":",
               linewidth=1, label=f"Vp_mean = {VP_MEAN * 100:.1f}%")
    ax.set_xlabel("Local Vp (%)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Through-thickness porosity profile at matched Vp_mean")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_fe_compare(fields, save_path):
    """Compare FE stiffness-retention along the midplane for uniform vs clustered."""
    uni_mesh, _ = fields["uniform"]
    clu_mesh, _ = fields["clustered (midplane)"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for label, mesh, color in (("uniform", uni_mesh, "tab:blue"),
                                ("clustered (midplane)", clu_mesh,
                                 "tab:orange")):
        # Slice nodes near the y midplane to get a (x, z) cross-section.
        y_mid = mesh.L_y / 2.0
        mask = np.isclose(mesh.nodes[:, 1], y_mid, atol=mesh.L_y / mesh.ny)
        z = mesh.nodes[mask, 2]
        sr = mesh.stiffness_reduction[mask]
        # Average across x at each unique z so the curve is 1-D.
        z_unique = np.unique(z)
        sr_avg = np.array([sr[np.isclose(z, zu)].mean() for zu in z_unique])
        ax.plot(sr_avg * 100.0, z_unique, label=label, color=color,
                linewidth=2)
    ax.set_xlabel("FE stiffness retention (1 - Vp_local), %")
    ax.set_ylabel("z (mm)")
    ax.set_title("FE-recovered midplane stiffness retention "
                 f"(matched Vp_mean = {VP_MEAN * 100:.1f}%)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    material = MATERIALS["T800_epoxy"]
    print(f"Distribution comparison at matched Vp_mean = "
          f"{VP_MEAN * 100:.2f}%  on T800_epoxy")
    print("=" * 78)

    empirical_rows = _empirical_table(material)
    print("\nEmpirical knockdowns (distribution x mode x model)")
    _print_table(empirical_rows, group_by_mode=True)

    fe_rows, fields = _fe_table(material)
    print("\nFE knockdowns (Tsai-Wu, two porosity-sensitive modes)")
    _print_table(fe_rows, group_by_mode=True)

    profiles_png = os.path.join(OUT_DIR, "distribution_profiles.png")
    _plot_profiles(material, profiles_png)
    print(f"\nPNG saved: {profiles_png}")

    fe_compare_png = os.path.join(OUT_DIR, "distribution_fe_compare.png")
    _plot_fe_compare(fields, fe_compare_png)
    print(f"PNG saved: {fe_compare_png}")

    print("\nFinding: empirical knockdowns are identical across all four "
          "distributions\nbecause EmpiricalSolver.get_failure_load uses the "
          "specimen-mean Vp; FE\nknockdowns differ because the field solve "
          "sees the local Vp_peak.")


if __name__ == "__main__":
    main()
