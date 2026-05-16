"""Porosity FE Analysis — Streamlit web app.

Browser front end for `porosity_fe_analysis`. Mirrors the structure used by
WrinkleFE: a sidebar of inputs feeds a cached analysis function whose result
fans out to Profile / Mesh / Results / Stress / Export tabs.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

# Make sibling modules importable when launched via `streamlit run app.py`
# from a checkout that hasn't been `pip install -e .`'d.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from porosity_fe_analysis import (
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FESolver,
    PorosityField,
)


# ======================================================================
# Pure helpers (kept module-level so tests can import them without Streamlit)
# ======================================================================

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_layup(text: str) -> list:
    """Parse a layup string like '[0/45/-45/90]_3s' to a flat angle list.

    Raises ValueError on malformed input rather than silently substituting
    a default — see issue #9.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Layup string is empty.")

    cleaned = text.replace("[", "").replace("]", "")

    symmetric = cleaned.endswith("s")
    if symmetric:
        cleaned = cleaned[:-1]

    repeat = 1
    if "_" in cleaned:
        cleaned, repeat_token = cleaned.rsplit("_", 1)
        try:
            repeat = int(repeat_token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid repeat count {repeat_token!r} in layup {text!r}; "
                f"expected an integer after the '_' (e.g. '[0/90]_3s')."
            ) from exc
        if repeat < 1:
            raise ValueError(
                f"Repeat count must be >= 1, got {repeat} in layup {text!r}."
            )

    sep = "/" if "/" in cleaned else ","
    tokens = [a.strip() for a in cleaned.split(sep) if a.strip()]
    if not tokens:
        raise ValueError(f"No ply angles found in layup {text!r}.")
    try:
        angles = [float(a) for a in tokens]
    except ValueError as exc:
        bad = next((a for a in tokens if not _is_float(a)), tokens[0])
        raise ValueError(
            f"Invalid ply angle {bad!r} in layup {text!r}; "
            f"expected numeric degrees (e.g. '[0/45/-45/90]_3s')."
        ) from exc

    angles = angles * repeat
    if symmetric:
        angles = angles + list(reversed(angles))
    return angles


def build_export_payload(result: dict) -> dict:
    """Flatten an analysis result into the export payload structure.

    Shared by the JSON and CSV writers so both formats describe the same
    fields. ``result`` is the dict produced by ``run_analysis``.
    """
    cfg = result["config"]
    emp = result["empirical"]
    payload = {
        "config": {
            "material": cfg["material_name"],
            "n_plies": cfg["n_plies"],
            "t_ply": cfg["t_ply"],
            "Vp_percent": cfg["Vp"],
            "distribution": cfg["distribution"],
            "void_shape": cfg["void_shape"],
            "mesh": f"{cfg['nx']}x{cfg['ny']}x{cfg['nz']}",
        },
        "empirical": {},
    }
    for mode in emp:
        payload["empirical"][mode] = {}
        for model in emp[mode]:
            r = emp[mode][model]
            payload["empirical"][mode][model] = {
                "failure_stress_MPa": r["failure_stress"],
                "knockdown": r["knockdown"],
            }
    return payload


def write_results_json(filepath: str, payload: dict) -> None:
    from porosity_fe_analysis import (
        FORMAT_EMPIRICAL_SWEEP, JSON_SCHEMA_VERSION,
        _build_provenance, _json_default,
    )
    envelope = {
        "schema_version": JSON_SCHEMA_VERSION,
        "format": FORMAT_EMPIRICAL_SWEEP,
        "provenance": _build_provenance(),
        **payload,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, default=_json_default)


def write_results_csv(filepath: str, payload: dict) -> None:
    """Write the export payload as a flat CSV.

    Configuration metadata is written as comment lines prefixed with ``#``;
    pandas (``read_csv(comment='#')``) and most CSV viewers handle this
    cleanly, while Excel ignores the comments and treats the table as data.
    """
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        for key, value in payload["config"].items():
            f.write(f"# {key}: {value}\n")
        writer = csv.writer(f)
        writer.writerow(["mode", "model", "failure_stress_MPa", "knockdown"])
        for mode in payload["empirical"]:
            for model in payload["empirical"][mode]:
                r = payload["empirical"][mode][model]
                writer.writerow([
                    mode, model,
                    r["failure_stress_MPa"],
                    r["knockdown"],
                ])


def _serialise_payload_json(payload: dict) -> str:
    from porosity_fe_analysis import (
        FORMAT_EMPIRICAL_SWEEP, JSON_SCHEMA_VERSION,
        _build_provenance, _json_default,
    )
    envelope = {
        "schema_version": JSON_SCHEMA_VERSION,
        "format": FORMAT_EMPIRICAL_SWEEP,
        "provenance": _build_provenance(),
        **payload,
    }
    return json.dumps(envelope, indent=2, default=_json_default)


def _serialise_payload_csv(payload: dict) -> str:
    buf = io.StringIO()
    for key, value in payload["config"].items():
        buf.write(f"# {key}: {value}\n")
    writer = csv.writer(buf)
    writer.writerow(["mode", "model", "failure_stress_MPa", "knockdown"])
    for mode in payload["empirical"]:
        for model in payload["empirical"][mode]:
            r = payload["empirical"][mode][model]
            writer.writerow([
                mode, model,
                r["failure_stress_MPa"],
                r["knockdown"],
            ])
    return buf.getvalue()


# ======================================================================
# Analysis runner (cached)
# ======================================================================

# Tuple ordering for the cache key. Keeping it explicit makes the cache
# invalidate cleanly when a new field is added.
_CFG_KEYS = (
    "material_name", "angles", "n_plies", "t_ply", "Vp",
    "distribution", "cluster_location", "void_shape", "loading_mode",
    "nx", "ny", "nz",
)


def _config_to_key(cfg: dict) -> tuple:
    return tuple((k, tuple(cfg[k]) if isinstance(cfg[k], list) else cfg[k])
                 for k in _CFG_KEYS)


@st.cache_data(show_spinner=False)
def run_analysis_cached(cfg_key: tuple) -> dict:
    """Cached wrapper around :func:`run_analysis`. ``cfg_key`` must be hashable."""
    cfg = {}
    for k, v in cfg_key:
        cfg[k] = list(v) if isinstance(v, tuple) else v
    return run_analysis(cfg)


def run_analysis(cfg: dict) -> dict:
    """Run the porosity analysis for one configuration.

    Returns a dict with keys: config, material, porosity_field, mesh,
    empirical, fe_field, fe_loading, fe_skipped_reason, f_md.
    """
    if cfg["material_name"] not in MATERIALS:
        raise ValueError(
            f"Unknown material {cfg['material_name']!r}. "
            f"Available presets: {sorted(MATERIALS)}."
        )
    material = MATERIALS[cfg["material_name"]]
    material = dataclasses.replace(
        material, t_ply=cfg["t_ply"], n_plies=cfg["n_plies"],
    )

    pf_kwargs = {
        "distribution": cfg["distribution"],
        "void_shape": cfg["void_shape"],
    }
    if cfg["distribution"] == "clustered":
        pf_kwargs["cluster_location"] = cfg["cluster_location"]

    porosity_field = PorosityField(material, cfg["Vp"] / 100.0, **pf_kwargs)

    mesh = CompositeMesh(
        porosity_field, material,
        nx=cfg["nx"], ny=cfg["ny"], nz=cfg["nz"],
        ply_angles=cfg["angles"],
    )

    empirical = EmpiricalSolver(mesh, material, ply_angles=cfg["angles"])
    emp_results = empirical.get_all_failure_loads()

    # FESolver BCs don't support ILSS short-beam shear today; skip the FE
    # pass entirely for ILSS so we don't silently substitute compression.
    loading_mode = cfg["loading_mode"]
    fe_supported = loading_mode in ("compression", "tension", "shear")

    fe_field = None
    fe_loading = None
    if fe_supported:
        applied_strain = -0.01 if loading_mode == "compression" else 0.01
        fe_loading = loading_mode
        fe_solver = FESolver(
            mesh, material, porosity_field, ply_angles=cfg["angles"],
        )
        fe_field = fe_solver.solve(
            loading=fe_loading, applied_strain=applied_strain, verbose=False,
        )

    return {
        "config": cfg,
        "material": material,
        "porosity_field": porosity_field,
        "mesh": mesh,
        "empirical": emp_results,
        "fe_field": fe_field,
        "fe_loading": fe_loading,
        "fe_skipped_reason": (
            None if fe_supported
            else f"FE solver does not support '{loading_mode}' boundary conditions"
        ),
        "f_md": empirical.f_md,
    }


# ======================================================================
# Plot routines (return a matplotlib Figure for st.pyplot)
# ======================================================================

def plot_profile(result: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    pf = result["porosity_field"]
    z, Vp = pf.effective_porosity_profile(nz=200)
    ax.plot(Vp * 100, z, "b-", linewidth=2)
    ax.set_xlabel("Porosity (%)", fontsize=11)
    ax.set_ylabel("z (mm)", fontsize=11)
    ax.set_title("Through-Thickness Porosity Profile",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    return fig


def plot_mesh(result: dict):
    """Mid-y cross-section coloured by stiffness retention with void overlays."""
    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = result["mesh"]
    nx1 = mesh.nx + 1
    ny1 = mesh.ny + 1
    ny_mid = mesh.ny // 2

    indices = []
    for k in range(mesh.nz + 1):
        for i in range(mesh.nx + 1):
            indices.append(k * ny1 * nx1 + ny_mid * nx1 + i)
    indices = np.array(indices)

    X = mesh.nodes[indices, 0].reshape(mesh.nz + 1, mesh.nx + 1)
    Z = mesh.nodes[indices, 2].reshape(mesh.nz + 1, mesh.nx + 1)
    Sr = mesh.stiffness_reduction[indices].reshape(mesh.nz + 1, mesh.nx + 1)

    im = ax.contourf(X, Z, Sr * 100, levels=20, cmap="viridis",
                     vmin=max(0, Sr.min() * 100 - 1), vmax=100)
    fig.colorbar(im, ax=ax, label="Stiffness Retention (%)")

    step_x = max(1, mesh.nx // 20)
    step_z = max(1, mesh.nz // 20)
    for k in range(0, mesh.nz + 1, step_z):
        row_x = mesh.nodes[
            [k * ny1 * nx1 + ny_mid * nx1 + i for i in range(mesh.nx + 1)], 0]
        row_z = mesh.nodes[
            [k * ny1 * nx1 + ny_mid * nx1 + i for i in range(mesh.nx + 1)], 2]
        ax.plot(row_x, row_z, "k-", linewidth=0.3, alpha=0.4)
    for i in range(0, mesh.nx + 1, step_x):
        col_x = mesh.nodes[
            [k * ny1 * nx1 + ny_mid * nx1 + i for k in range(mesh.nz + 1)], 0]
        col_z = mesh.nodes[
            [k * ny1 * nx1 + ny_mid * nx1 + i for k in range(mesh.nz + 1)], 2]
        ax.plot(col_x, col_z, "k-", linewidth=0.3, alpha=0.4)

    void_elems = mesh.void_elements
    if len(void_elems) > 0:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        void_patches = []
        for e_idx in void_elems:
            j_e = (e_idx // mesh.nx) % mesh.ny
            if j_e != mesh.ny // 2:
                continue
            node_coords = mesh.nodes[mesh.elements[e_idx]]
            xz = node_coords[:, [0, 2]]
            unique_xz = np.unique(xz, axis=0)
            if len(unique_xz) < 3:
                continue
            cx_p, cz_p = unique_xz.mean(axis=0)
            angles = np.arctan2(
                unique_xz[:, 1] - cz_p, unique_xz[:, 0] - cx_p,
            )
            order = np.argsort(angles)
            void_patches.append(Polygon(unique_xz[order], closed=True))

        if void_patches:
            pc = PatchCollection(
                void_patches, facecolor="white", edgecolor="red",
                linewidth=1.0, zorder=5, alpha=1.0,
            )
            ax.add_collection(pc)
            ax.plot([], [], "s", color="white", markeredgecolor="red",
                    markeredgewidth=1.0,
                    label=f"Voids ({len(void_patches)})")
            ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("z (mm)", fontsize=11)
    ax.set_title(
        f"FE Mesh — Stiffness Retention  |  "
        f"{len(mesh.nodes):,} nodes, {len(mesh.elements):,} elements",
        fontsize=12, fontweight="bold",
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_results(result: dict, layup_str: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    emp = result["empirical"]
    fe_field = result.get("fe_field")
    fe_loading = result.get("fe_loading", "compression")
    cfg = result["config"]
    f_md = result.get("f_md", 0.5)

    modes = ["compression", "tension", "shear", "ilss"]
    models = ["judd_wright", "power_law", "linear"]
    model_labels = ["Judd-Wright", "Power Law", "Linear"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    hatches = [None, None, None]

    has_fe = fe_field is not None
    if has_fe:
        models.append("fe")
        model_labels.append(f"FE Stiffness ({fe_loading})")
        colors.append("#d62728")
        hatches.append("//")

    n_models = len(models)
    x = np.arange(len(modes))
    width = 0.8 / n_models

    for i, (model_key, label, color, hatch) in enumerate(
        zip(models, model_labels, colors, hatches)
    ):
        vals = []
        for mode in modes:
            if model_key == "fe":
                vals.append(fe_field.knockdown if mode == fe_loading else float("nan"))
            else:
                vals.append(emp[mode][model_key]["knockdown"])
        bar_x = x + i * width - (n_models - 1) * width / 2
        for bx, bv in zip(bar_x, vals):
            if not np.isnan(bv):
                ax.bar(bx, bv, width, color=color, hatch=hatch,
                       edgecolor="white" if hatch is None else "0.3",
                       label=label if bx == bar_x[0] else "")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes], fontsize=10)
    ax.set_ylabel("Knockdown Factor", fontsize=11)
    ax.set_title(
        f"Knockdown Factor by Loading Mode  |  "
        f"Vp = {cfg['Vp']:.1f}%, {cfg['void_shape']}, "
        f"{cfg['distribution']}, {layup_str}",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")

    note = ("Solid bars = strength knockdown (at mean Vp); "
            "hatched bar = stiffness knockdown (FE)")
    if f_md < 0.49:
        note += (f"\nLayup scaling: f_md = {f_md:.2f} "
                 "(coefficients reduced for fiber-dominated layup)")
    ax.text(0.01, 0.01, note, transform=ax.transAxes,
            fontsize=7, color="0.4", va="bottom")

    fig.tight_layout()
    return fig


_STRESS_COMPONENTS = {
    "σ₁₁ (fiber)": (0, r"$\sigma_{11}$ local (MPa)"),
    "σ₂₂ (transverse)": (1, r"$\sigma_{22}$ local (MPa)"),
    "σ₃₃ (through-thickness)": (2, r"$\sigma_{33}$ local (MPa)"),
    "τ₂₃ (interlaminar)": (3, r"$\tau_{23}$ local (MPa)"),
    "τ₁₃ (interlaminar)": (4, r"$\tau_{13}$ local (MPa)"),
    "τ₁₂ (in-plane shear)": (5, r"$\tau_{12}$ local (MPa)"),
    "Von Mises": (-1, "Von Mises Stress (MPa)"),
}


def plot_stress(result: dict, comp_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    fe_field = result.get("fe_field")
    if fe_field is None:
        ax.text(0.5, 0.5, "No FE results available.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="0.5")
        ax.set_axis_off()
        return fig

    mesh = result["mesh"]
    stress_local = fe_field.stress_local
    comp_idx, label = _STRESS_COMPONENTS.get(comp_name, (0, comp_name + " (MPa)"))

    if comp_idx == -1:
        s = stress_local.mean(axis=1)
        s1, s2, s3 = s[:, 0], s[:, 1], s[:, 2]
        s4, s5, s6 = s[:, 3], s[:, 4], s[:, 5]
        elem_stress = np.sqrt(0.5 * (
            (s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2
            + 6.0 * (s4 ** 2 + s5 ** 2 + s6 ** 2)
        ))
    else:
        elem_stress = stress_local.mean(axis=1)[:, comp_idx]

    ny_mid = mesh.ny // 2
    mid_elem_indices = []
    for k in range(mesh.nz):
        for i in range(mesh.nx):
            mid_elem_indices.append(k * mesh.ny * mesh.nx + ny_mid * mesh.nx + i)
    mid_elem_indices = np.array(mid_elem_indices)

    elem_nodes_coords = mesh.nodes[mesh.elements[mid_elem_indices]]
    cx = elem_nodes_coords[:, :, 0].mean(axis=1)
    cz = elem_nodes_coords[:, :, 2].mean(axis=1)

    interior_mask = (cx > mesh.L_x * 0.10) & (cx < mesh.L_x * 0.90)
    mid_elem_indices = mid_elem_indices[interior_mask]
    cx = cx[interior_mask]
    cz = cz[interior_mask]
    sv = elem_stress[mid_elem_indices]

    finite_mask = np.isfinite(sv)
    if finite_mask.sum() >= 3:
        # Symmetric range so RdBu_r's white midpoint is true σ=0; using raw
        # 5/95 percentiles shifts the neutral color off zero and makes the
        # sign visually misread (#51).
        p5 = np.percentile(sv[finite_mask], 5)
        p95 = np.percentile(sv[finite_mask], 95)
        v = max(abs(p5), abs(p95)) or 1.0
        tcf = ax.tricontourf(cx[finite_mask], cz[finite_mask],
                             sv[finite_mask], levels=20, cmap="RdBu_r",
                             vmin=-v, vmax=v)
        fig.colorbar(tcf, ax=ax, label=label)
    else:
        ax.text(0.5, 0.5, "Insufficient interior data for contour plot.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.5")

    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("z (mm)", fontsize=11)
    ax.set_title(
        f"FE Stress (local/material frame): {comp_name}  |  "
        "interior, mid-y cross-section",
        fontsize=11, fontweight="bold",
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ======================================================================
# Streamlit UI
# ======================================================================

_DISTRIBUTION_OPTIONS = {
    "uniform": ("uniform", "midplane"),
    "clustered (midplane)": ("clustered", "midplane"),
    "clustered (surface)": ("clustered", "surface"),
    "interface": ("interface", "midplane"),
}


def _render():
    st.set_page_config(
        page_title="PorosityFE",
        page_icon=None,
        layout="wide",
    )
    st.title("PorosityFE — Composite Laminate Porosity Analysis")
    st.caption(
        "Predict strength and stiffness knockdown in porosity-degraded "
        "composite laminates. Adjust inputs in the sidebar, then click **Run analysis**."
    )

    # ---- Sidebar inputs ----------------------------------------------------
    with st.sidebar:
        st.header("Inputs")
        expert = st.toggle(
            "Expert mode",
            value=False,
            help="Show mesh resolution sliders and other advanced options.",
        )

        st.subheader("Material & laminate")
        material_name = st.selectbox(
            "Material",
            options=list(MATERIALS.keys()),
            index=0,
        )
        layup_str = st.text_input(
            "Layup",
            value="[0/45/-45/90]_3s",
            help=(
                "Ply angles separated by '/'. Use '_Ns' for N repeats and "
                "trailing 's' for symmetric. Examples: [0/45/-45/90]_3s, "
                "[0/90]_6s, 0/0/0/90/90/90."
            ),
        )
        t_ply = st.number_input(
            "Ply thickness (mm)",
            min_value=0.05, max_value=0.50, value=0.183, step=0.01, format="%.3f",
        )

        st.subheader("Porosity")
        Vp = st.number_input(
            "Void volume fraction (%)",
            min_value=0.1, max_value=15.0, value=3.0, step=0.5, format="%.1f",
            help=(
                "Typical range: 0.5–5% for autoclave, 2–10% for OOA."
            ),
        )
        distribution_label = st.selectbox(
            "Distribution",
            options=list(_DISTRIBUTION_OPTIONS.keys()),
            index=0,
            help=(
                "uniform: constant porosity throughout.\n"
                "clustered (midplane): Gaussian peak at midplane.\n"
                "clustered (surface): Gaussian peak at surface.\n"
                "interface: concentrated at ply interfaces."
            ),
        )
        void_shape = st.selectbox(
            "Void shape",
            options=["spherical", "cylindrical", "penny"],
            index=0,
            help=(
                "spherical: equiaxed (AR=1)\n"
                "cylindrical: prolate (AR=3)\n"
                "penny: oblate disc (AR=10)"
            ),
        )

        st.subheader("Loading")
        loading_mode = st.selectbox(
            "Loading mode",
            options=["compression", "tension", "shear", "ilss"],
            index=0,
            help=(
                "All four modes are computed empirically; this selects the "
                "primary mode for the FE solve and bar-chart highlight. "
                "ILSS skips the FE pass (BCs not supported)."
            ),
        )

        st.subheader("Mesh")
        if expert:
            nx = st.slider("nx", min_value=2, max_value=200, value=30, step=1)
            ny = st.slider("ny", min_value=2, max_value=100, value=10, step=1)
            nz = st.slider("nz", min_value=2, max_value=100, value=12, step=1)
        else:
            nx, ny, nz = 30, 10, 12
            st.caption(f"Default mesh: {nx} × {ny} × {nz} (enable Expert mode to change).")

        run = st.button("Run analysis", type="primary", use_container_width=True)

    # ---- Build config from sidebar state -----------------------------------
    try:
        angles = parse_layup(layup_str)
    except ValueError as exc:
        st.error(f"Invalid layup: {exc}")
        return

    distribution, cluster_location = _DISTRIBUTION_OPTIONS[distribution_label]
    cfg = {
        "material_name": material_name,
        "angles": angles,
        "n_plies": len(angles),
        "t_ply": float(t_ply),
        "Vp": float(Vp),
        "distribution": distribution,
        "cluster_location": cluster_location,
        "void_shape": void_shape,
        "loading_mode": loading_mode,
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
    }

    # ---- Run analysis (only when the button is pressed) --------------------
    if run:
        with st.spinner("Running porosity analysis…"):
            try:
                st.session_state["result"] = run_analysis_cached(_config_to_key(cfg))
                st.session_state["layup_str"] = layup_str
            except Exception as exc:
                logger.exception("Analysis failed")
                st.session_state["result"] = None
                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("result")
    layup_for_title = st.session_state.get("layup_str", layup_str)

    # ---- Tabs --------------------------------------------------------------
    tab_overview, tab_profile, tab_mesh, tab_results, tab_stress, tab_export = st.tabs(
        ["Overview", "Profile", "Mesh", "Results", "Stress", "Export"]
    )

    with tab_overview:
        st.markdown(
            """
            **PorosityFE** estimates strength and stiffness knockdown in
            porosity-degraded composite laminates using empirical
            models (Judd–Wright, power law, linear) and a 3D hex finite-element
            solve. Configure the laminate and porosity field in the sidebar and
            press **Run analysis**.

            - **Profile** — through-thickness porosity distribution
            - **Mesh** — mid-y cross-section of the FE mesh, coloured by stiffness retention
            - **Results** — empirical knockdown bar chart with the FE stiffness knockdown overlaid
            - **Stress** — FE stress contour for a chosen component (skipped for ILSS)
            - **Export** — download the empirical knockdown sweep as JSON or CSV
            """
        )
        if result is None:
            st.info("No results yet. Adjust the sidebar and press **Run analysis**.")
        else:
            cfg_r = result["config"]
            st.success(
                f"Last run: {cfg_r['material_name']}, layup {layup_for_title}, "
                f"Vp = {cfg_r['Vp']:.1f}%, {cfg_r['void_shape']}, "
                f"{cfg_r['distribution']}, mesh {cfg_r['nx']}×{cfg_r['ny']}×{cfg_r['nz']}."
            )
            if result.get("fe_skipped_reason"):
                st.warning(f"FE skipped: {result['fe_skipped_reason']}")

    def _placeholder():
        st.info("Run an analysis to populate this tab.")

    with tab_profile:
        if result is None:
            _placeholder()
        else:
            st.pyplot(plot_profile(result), clear_figure=True)

    with tab_mesh:
        if result is None:
            _placeholder()
        else:
            st.pyplot(plot_mesh(result), clear_figure=True)

    with tab_results:
        if result is None:
            _placeholder()
        else:
            st.pyplot(plot_results(result, layup_for_title), clear_figure=True)

    with tab_stress:
        if result is None:
            _placeholder()
        elif result.get("fe_field") is None:
            st.warning(
                result.get("fe_skipped_reason")
                or "No FE field available for this configuration."
            )
        else:
            comp_name = st.selectbox(
                "Stress component",
                options=list(_STRESS_COMPONENTS.keys()),
                index=0,
            )
            st.pyplot(plot_stress(result, comp_name), clear_figure=True)

    with tab_export:
        if result is None:
            _placeholder()
        else:
            payload = build_export_payload(result)
            st.download_button(
                "Download JSON",
                data=_serialise_payload_json(payload),
                file_name="porosity_results.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                "Download CSV",
                data=_serialise_payload_csv(payload),
                file_name="porosity_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            with st.expander("Preview JSON"):
                st.code(_serialise_payload_json(payload), language="json")


try:
    from streamlit.runtime import exists as _st_runtime_exists
    _UNDER_STREAMLIT = _st_runtime_exists()
except Exception:
    _UNDER_STREAMLIT = False

if _UNDER_STREAMLIT:
    _render()
