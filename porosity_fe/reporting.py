"""Reporting helpers shared by ``app.py`` and the test suite.

These functions used to live at module scope inside ``app.py``, but ``app.py``
imports ``streamlit`` at module load. Pulling the streamlit-free helpers into
a dedicated module lets the core test matrix import them (``from
porosity_fe.reporting import build_ncr_record``) without dragging in the
optional ``web`` extra (#155).

Two layers of functionality live here:

* The export payload writers (JSON / CSV) used by the Results / Export tabs.
* The Nonconformance Report (NCR) builder, plus its JSON / Markdown / PDF
  serialisers, which support the MRB disposition workflow.

The module deliberately keeps its top-level imports cheap. ``matplotlib`` is
only needed by :func:`serialise_ncr_pdf`, so it is imported lazily inside
that function.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import textwrap


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
    from porosity_fe import (
        FORMAT_EMPIRICAL_SWEEP,
        JSON_SCHEMA_VERSION,
        _build_provenance,
        _json_default,
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
    from porosity_fe import (
        FORMAT_EMPIRICAL_SWEEP,
        JSON_SCHEMA_VERSION,
        _build_provenance,
        _json_default,
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
# Nonconformance Report (NCR) — MRB disposition support
# ======================================================================
#
# Lets a field engineer turn a porosity analysis into a structured NCR that
# an MRB can review. The tool produces *analysis and a recommended
# disposition path* — it never issues a final disposition. The governing
# (worst-case) knockdown across all modes/models drives the recommendation,
# because the MRB cares about the most-critical residual strength, not the
# average.

STRUCTURAL_CLASSES = ("primary", "secondary", "non-structural")


def governing_failure(result: dict) -> dict:
    """Most-critical empirical case (lowest knockdown) across modes/models.

    The MRB evaluates the worst-case residual strength, so the
    disposition recommendation keys off the minimum knockdown rather than
    a per-mode average.
    """
    emp = result["empirical"]
    worst = None
    for mode in emp:
        for model in emp[mode]:
            r = emp[mode][model]
            kd = r["knockdown"]
            if worst is None or kd < worst["knockdown"]:
                worst = {
                    "mode": mode,
                    "model": model,
                    "knockdown": kd,
                    "residual_strength_MPa": r["failure_stress"],
                }
    if worst is None:
        raise ValueError("Analysis result has no empirical knockdown data.")
    return worst


def recommend_disposition(
    Vp: float, governing_knockdown: float, structural_class: str = "primary"
) -> dict:
    """Recommend (not decide) an MRB disposition path for a porosity NCR.

    The recommendation bins on the measured void content and the governing
    (worst-case) knockdown, then escalates required substantiation by
    structural class. It is deliberately conservative: the released
    engineering drawing / process spec is the governing acceptance
    authority, and the MRB must substantiate against it.
    """
    structural_class = structural_class if structural_class in STRUCTURAL_CLASSES else "primary"

    if Vp <= 1.0 and governing_knockdown >= 0.95:
        path = "Use-As-Is (UAI) — pending MRB concurrence"
        rationale = (
            f"Measured void content ({Vp:.2f}%) is within the porosity range "
            f"typical of autoclave-grade primary structure, and the predicted "
            f"governing residual strength retains "
            f"{governing_knockdown * 100:.1f}% of pristine. Strength loss is "
            f"minor and likely covered by as-designed margin."
        )
    elif Vp <= 2.0 and governing_knockdown >= 0.90:
        path = "Use-As-Is with Engineering Evaluation"
        rationale = (
            f"Void content ({Vp:.2f}%) is marginal against typical drawing "
            f"allowables and the governing knockdown "
            f"({governing_knockdown * 100:.1f}% retained) indicates moderate "
            f"strength loss. UAI may be substantiated by a positive "
            f"margin-of-safety check at the knockdown-adjusted allowable."
        )
    elif Vp <= 5.0 and governing_knockdown >= 0.80:
        path = "Engineering Evaluation / Repair"
        rationale = (
            f"Void content ({Vp:.2f}%) exceeds porosity allowables typical of "
            f"primary structure and the governing knockdown leaves only "
            f"{governing_knockdown * 100:.1f}% of pristine strength. "
            f"Disposition hinges on the as-designed margin and on whether an "
            f"approved repair can restore a serviceable condition."
        )
    else:
        path = "Repair or Scrap"
        rationale = (
            f"Void content ({Vp:.2f}%) and the predicted governing knockdown "
            f"({governing_knockdown * 100:.1f}% retained) represent a severe "
            f"degradation. Use-As-Is is not recommended without exceptional, "
            f"test-backed substantiation; repair or scrap is the expected path."
        )

    cited_criteria = [
        "Released engineering drawing / part specification porosity allowable "
        "(governing acceptance limit — verify against the controlled drawing).",
        "Process specification cure / void-content requirements for the "
        "applicable material system.",
        "Structural substantiation: laminate margin of safety recomputed at "
        "the knockdown-adjusted allowable for the governing loading mode.",
        "NDI acceptance criteria (ultrasonic C-scan attenuation / void "
        "content) per the applicable NDT specification.",
        "Quality-system MRB procedure for disposition of nonconforming "
        "material.",
    ]

    required_mrb_actions = [
        "Confirm measured void content by micrograph / acid digestion or a "
        "validated C-scan correlation.",
        "Verify the governing porosity allowable on the released engineering "
        "drawing / specification.",
        "Perform structural substantiation: recompute the margin of safety "
        "using the knockdown-adjusted allowable for the governing mode.",
        "Define the NDI extent and map the affected zone / part region.",
    ]
    if structural_class == "primary":
        required_mrb_actions.append(
            "Primary structure: obtain customer / DER engineering concurrence "
            "before approving any Use-As-Is disposition."
        )
    elif structural_class == "non-structural":
        required_mrb_actions.append(
            "Non-structural item: confirm there is no fluid-ingress, "
            "fatigue, or interface-sealing function affected by the porosity."
        )
    if path.startswith("Repair") or "Repair" in path:
        required_mrb_actions.append(
            "If repair is selected, document the approved repair scheme and "
            "the post-repair re-inspection requirements."
        )

    disclaimer = (
        "This recommended disposition path was produced by an automated MRB "
        "support tool from a predictive porosity-knockdown analysis. It is "
        "NOT a final disposition. A qualified Material Review Board must "
        "independently review, may modify, and must formally approve the "
        "disposition. Final acceptance requires substantiation against the "
        "governing engineering drawing / specification and the applicable "
        "structural margins."
    )

    return {
        "path": path,
        "structural_class": structural_class,
        "rationale": rationale,
        "cited_criteria": cited_criteria,
        "required_mrb_actions": required_mrb_actions,
        "disclaimer": disclaimer,
    }


def build_ncr_record(result: dict, meta: dict) -> dict:
    """Build an analysis validation summary to attach to an NCR.

    This is the technical attachment, not a full NCR form: it carries the
    porosity analysis, the governing knockdown, and a recommended (not
    final) disposition path. Part/serial/work-order identification lives on
    the parent NCR — ``meta`` only needs who prepared it and an optional
    parent-NCR reference. Everything technical is derived from ``result`` so
    the summary cannot drift from what was actually run.
    """
    payload = build_export_payload(result)
    cfg = payload["config"]
    worst = governing_failure(result)
    Vp = float(cfg["Vp_percent"])
    structural_class = meta.get("structural_class", "primary")

    disposition = recommend_disposition(
        Vp, worst["knockdown"], structural_class
    )

    today = datetime.date.today().isoformat()
    layup = meta.get("layup") or "(see analysis configuration)"

    return {
        "summary": {
            "title": "Composite Porosity Analysis — NCR Validation Summary",
            "prepared_by": meta.get("prepared_by", ""),
            "ncr_reference": meta.get("ncr_reference", ""),
            "date": meta.get("date") or today,
            "structural_class": structural_class,
            "note": meta.get("note", ""),
        },
        "nonconformance": {
            "summary": (
                f"Porosity / void content of {Vp:.2f}% in a "
                f"{cfg['material']} laminate ({cfg['n_plies']} plies, layup "
                f"{layup}); {cfg['distribution']} distribution, "
                f"{cfg['void_shape']} void morphology. Predicted to exceed "
                f"typical drawing porosity allowables and to knock down "
                f"residual strength — see engineering analysis."
            ),
            "material": cfg["material"],
            "layup": layup,
            "n_plies": cfg["n_plies"],
            "t_ply_mm": cfg["t_ply"],
            "measured_Vp_percent": Vp,
            "distribution": cfg["distribution"],
            "void_shape": cfg["void_shape"],
            "analysis_mesh": cfg["mesh"],
        },
        "engineering_analysis": {
            "governing_mode": worst["mode"],
            "governing_model": worst["model"],
            "governing_knockdown": worst["knockdown"],
            "governing_residual_strength_MPa": worst["residual_strength_MPa"],
            "per_mode": payload["empirical"],
        },
        "recommended_disposition": disposition,
    }


def serialise_ncr_json(ncr: dict) -> str:
    from porosity_fe import (
        FORMAT_NCR,
        JSON_SCHEMA_VERSION,
        _build_provenance,
        _json_default,
    )
    envelope = {
        "schema_version": JSON_SCHEMA_VERSION,
        "format": FORMAT_NCR,
        "provenance": _build_provenance(),
        **ncr,
    }
    return json.dumps(envelope, indent=2, default=_json_default)


def _per_mode_rows(ea: dict) -> list:
    """Flatten the per-mode/model knockdown table to (mode, model, MPa, kd)."""
    rows = []
    for mode in ea["per_mode"]:
        for model in ea["per_mode"][mode]:
            r = ea["per_mode"][mode][model]
            rows.append((
                mode, model,
                f"{r['failure_stress_MPa']:.1f}",
                f"{r['knockdown']:.3f}",
            ))
    return rows


def serialise_ncr_markdown(ncr: dict) -> str:
    """Render the analysis validation summary as a Markdown attachment."""
    s = ncr["summary"]
    nc = ncr["nonconformance"]
    ea = ncr["engineering_analysis"]
    dp = ncr["recommended_disposition"]

    lines = []
    lines.append(f"# {s['title']}")
    lines.append("")
    lines.append("_Attachment to a Nonconformance Report — analysis validation_")
    lines.append("")
    lines.append(f"- Prepared by: {s['prepared_by'] or '—'}")
    lines.append(f"- Parent NCR reference: {s['ncr_reference'] or '—'}")
    lines.append(f"- Date: {s['date']}")
    lines.append(f"- Structural classification: {s['structural_class']}")
    if s.get("note"):
        lines.append(f"- Engineer note: {s['note']}")
    lines.append("")

    lines.append("## 1. Nonconformance Summary")
    lines.append("")
    lines.append(nc["summary"])
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Material | {nc['material']} |")
    lines.append(f"| Layup | {nc['layup']} |")
    lines.append(f"| Plies | {nc['n_plies']} |")
    lines.append(f"| Ply thickness (mm) | {nc['t_ply_mm']} |")
    lines.append(f"| Measured void content Vp (%) | {nc['measured_Vp_percent']:.2f} |")
    lines.append(f"| Distribution | {nc['distribution']} |")
    lines.append(f"| Void morphology | {nc['void_shape']} |")
    lines.append(f"| Analysis mesh | {nc['analysis_mesh']} |")
    lines.append("")

    lines.append("## 2. Engineering Analysis (predicted porosity knockdown)")
    lines.append("")
    lines.append(
        f"**Governing (worst-case) case:** {ea['governing_mode']} / "
        f"{ea['governing_model']} — knockdown "
        f"{ea['governing_knockdown']:.3f} "
        f"({ea['governing_knockdown'] * 100:.1f}% of pristine retained), "
        f"residual strength "
        f"{ea['governing_residual_strength_MPa']:.1f} MPa."
    )
    lines.append("")
    lines.append("| Mode | Model | Residual strength (MPa) | Knockdown |")
    lines.append("|---|---|---|---|")
    for mode, model, mpa, kd in _per_mode_rows(ea):
        lines.append(f"| {mode} | {model} | {mpa} | {kd} |")
    lines.append("")

    lines.append(
        "## 3. Recommended Disposition Path "
        "(for MRB review — NOT a final disposition)"
    )
    lines.append("")
    lines.append(f"**Recommended path:** {dp['path']}")
    lines.append("")
    lines.append(f"**Rationale:** {dp['rationale']}")
    lines.append("")
    lines.append(f"> {dp['disclaimer']}")
    lines.append("")

    lines.append("## 4. Cited Criteria")
    lines.append("")
    for c in dp["cited_criteria"]:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("## 5. Required MRB Actions")
    lines.append("")
    for a in dp["required_mrb_actions"]:
        lines.append(f"- [ ] {a}")
    lines.append("")
    lines.append(
        "_Generated by PorosityFE MRB support tool. This is a predictive "
        "analysis and a recommended disposition path; it does not constitute "
        "MRB approval._"
    )
    lines.append("")
    return "\n".join(lines)


_PDF_WRAP = 92
_PDF_LINES_PER_PAGE = 58


def _ncr_text_lines(ncr: dict) -> list:
    """Plain-text (no Markdown) lines for the PDF, wrapped to a fixed width."""
    s = ncr["summary"]
    nc = ncr["nonconformance"]
    ea = ncr["engineering_analysis"]
    dp = ncr["recommended_disposition"]

    out: list = []

    def para(text: str, indent: str = "") -> None:
        for w in textwrap.wrap(text, _PDF_WRAP - len(indent)) or [""]:
            out.append(indent + w)

    def rule() -> None:
        out.append("-" * _PDF_WRAP)

    out.append(s["title"].upper())
    out.append("Attachment to a Nonconformance Report - analysis validation")
    rule()
    out.append(f"Prepared by:               {s['prepared_by'] or '-'}")
    out.append(f"Parent NCR reference:      {s['ncr_reference'] or '-'}")
    out.append(f"Date:                      {s['date']}")
    out.append(f"Structural classification: {s['structural_class']}")
    if s.get("note"):
        para(f"Engineer note: {s['note']}")
    out.append("")

    out.append("1. NONCONFORMANCE SUMMARY")
    rule()
    para(nc["summary"])
    out.append("")
    out.append(f"  Material:                {nc['material']}")
    out.append(f"  Layup:                   {nc['layup']}")
    out.append(f"  Plies:                   {nc['n_plies']}")
    out.append(f"  Ply thickness (mm):      {nc['t_ply_mm']}")
    out.append(f"  Measured void Vp (%):    {nc['measured_Vp_percent']:.2f}")
    out.append(f"  Distribution:            {nc['distribution']}")
    out.append(f"  Void morphology:         {nc['void_shape']}")
    out.append(f"  Analysis mesh:           {nc['analysis_mesh']}")
    out.append("")

    out.append("2. ENGINEERING ANALYSIS (predicted porosity knockdown)")
    rule()
    para(
        f"Governing (worst-case) case: {ea['governing_mode']} / "
        f"{ea['governing_model']} - knockdown {ea['governing_knockdown']:.3f} "
        f"({ea['governing_knockdown'] * 100:.1f}% of pristine retained), "
        f"residual strength {ea['governing_residual_strength_MPa']:.1f} MPa."
    )
    out.append("")
    out.append(
        f"  {'Mode':<14}{'Model':<14}{'Resid. (MPa)':>14}{'Knockdown':>12}"
    )
    out.append("  " + "-" * 52)
    for mode, model, mpa, kd in _per_mode_rows(ea):
        out.append(f"  {mode:<14}{model:<14}{mpa:>14}{kd:>12}")
    out.append("")

    out.append("3. RECOMMENDED DISPOSITION PATH")
    out.append("   (for MRB review - NOT a final disposition)")
    rule()
    para(f"Recommended path: {dp['path']}")
    out.append("")
    para(f"Rationale: {dp['rationale']}")
    out.append("")
    para(dp["disclaimer"], indent="  ")
    out.append("")

    out.append("4. CITED CRITERIA")
    rule()
    for c in dp["cited_criteria"]:
        para(f"- {c}", indent="  ")
    out.append("")

    out.append("5. REQUIRED MRB ACTIONS")
    rule()
    for a in dp["required_mrb_actions"]:
        para(f"[ ] {a}", indent="  ")
    out.append("")
    para(
        "Generated by PorosityFE MRB support tool. This is a predictive "
        "analysis and a recommended disposition path; it does not constitute "
        "MRB approval."
    )
    return out


def serialise_ncr_pdf(ncr: dict) -> bytes:
    """Render the validation summary as a paginated US-Letter PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    lines = _ncr_text_lines(ncr)
    pages = [
        lines[i:i + _PDF_LINES_PER_PAGE]
        for i in range(0, len(lines), _PDF_LINES_PER_PAGE)
    ] or [[""]]

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for page in pages:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(
                0.07, 0.96, "\n".join(page),
                family="monospace", fontsize=8.5,
                va="top", ha="left",
            )
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()


def write_ncr_json(filepath: str, ncr: dict) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(serialise_ncr_json(ncr))


def write_ncr_markdown(filepath: str, ncr: dict) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(serialise_ncr_markdown(ncr))


def write_ncr_pdf(filepath: str, ncr: dict) -> None:
    with open(filepath, "wb") as f:
        f.write(serialise_ncr_pdf(ncr))
