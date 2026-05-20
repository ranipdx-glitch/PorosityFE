"""argparse-driven CLI (``porosity-analyze``)."""

import argparse
import logging
import os
import sys
from typing import List, Optional

from . import __version__
from .io import save_results_to_json
from .materials import MATERIALS
from .pipeline import compare_configurations
from .porosity_field import POROSITY_CONFIGS
from .viz import FEVisualizer

logger = logging.getLogger("porosity_fe_analysis")

# Default Vp sweep — preserves the historical hardcoded behavior so that
# `porosity-analyze` with no arguments reproduces today's analysis range.
DEFAULT_POROSITY_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.08]


def _vp_label(Vp: float) -> str:
    """Stable, filesystem-safe label for a void fraction.

    Integer-percent fractions keep the legacy ``Npct`` form (e.g. 0.03 ->
    ``3pct``); non-integer fractions fall back to a decimal-derived form
    (e.g. 0.025 -> ``2p5pct``) so distinct sweeps never collide.
    """
    pct = Vp * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}pct"
    return f"{pct:.4f}".rstrip('0').rstrip('.').replace('.', 'p') + "pct"


def _build_arg_parser() -> 'argparse.ArgumentParser':
    """Construct the argparse driver for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        prog="porosity-analyze",
        description=(
            "Run the porosity-degraded composite laminate analysis over one "
            "or more void volume fractions and write JSON results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--material",
        default="T800_epoxy",
        help="Material preset name (validated against the built-in presets).",
    )
    parser.add_argument(
        "--vp",
        type=float,
        nargs="+",
        default=list(DEFAULT_POROSITY_LEVELS),
        metavar="VP",
        help=(
            "One or more void volume fractions in [0, 1]. Defaults to the "
            "historical sweep."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write JSON results into (created if missing).",
    )
    parser.add_argument(
        "--applied-stress",
        type=float,
        default=-1500.0,
        help="Applied stress (MPa) passed to compare_configurations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Recorded in JSON provenance for reproducibility. The pipeline "
            "is deterministic (RNG-free), so this does not alter results."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Also render the heavy matplotlib figures (PNG). Off by default "
            "to keep CI / batch runs fast."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress progress output; only warnings and errors are shown. "
            "Mutually exclusive with --verbose."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Show debug-level progress output in addition to the default "
            "INFO progress. Mutually exclusive with --quiet."
        ),
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="List the available material presets and exit.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of worker processes for the per-configuration sweep "
            "in compare_configurations (#52). 1 (default) runs serially "
            "(deterministic, byte-identical to legacy behaviour); N>1 "
            "parallelises the (Vp, config) calls across N processes; 0 or "
            "-1 uses os.cpu_count(). Results are deterministically "
            "re-assembled regardless of N."
        ),
    )
    return parser


class _DynamicStdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stdout`` on every emit.

    Plain ``StreamHandler(sys.stdout)`` caches the stream reference at
    construction time, which breaks pytest's ``capsys`` fixture (it
    rebinds ``sys.stdout`` per-test). Looking it up lazily keeps the
    formatting identical to the old bare-``print`` output while
    remaining capturable.
    """

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value):
        # logging.StreamHandler.__init__ assigns ``self.stream`` -- accept
        # and ignore so the dynamic property above wins.
        pass


def _configure_cli_logging(*, quiet: bool, verbose: bool) -> None:
    """Wire the module ``logger`` for CLI use.

    Routes progress through the module logger so ``--quiet`` actually
    silences the run (issue #78). Attaches a stdout stream handler the
    first time the CLI is invoked so the output looks like the old
    bare-``print`` style.

    Parameters
    ----------
    quiet : bool
        Suppress INFO/DEBUG; only warnings and errors are surfaced.
    verbose : bool
        Lower the threshold to DEBUG.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Don't propagate to the root logger -- pytest's caplog and library
    # consumers (the Streamlit app) configure their own handlers and we
    # don't want duplicated lines.
    logger.propagate = False
    logger.setLevel(level)

    # Reuse any handler we already attached; otherwise add a simple
    # stdout stream so the formatting matches the old print()-based UX.
    if not any(getattr(h, "_porosity_cli", False) for h in logger.handlers):
        handler = _DynamicStdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._porosity_cli = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def _resolve_via_shim(name: str, fallback):
    """Look up ``name`` on the back-compat ``porosity_fe_analysis`` shim if
    it has been imported (so test ``monkeypatch.setattr`` on the shim is
    honoured), otherwise return ``fallback``.

    The shim re-exports the package's public surface, so attribute lookup
    on it usually returns the same object that lives in the package. Tests
    however historically patch the shim directly (``monkeypatch.setattr(
    porosity_fe_analysis, 'compare_configurations', spy)``) and expect the
    CLI to honour that patch. Going through the shim preserves that
    contract after the #119 split.
    """
    import sys as _sys
    shim = _sys.modules.get('porosity_fe_analysis')
    if shim is None:
        return fallback
    return getattr(shim, name, fallback)


def main(argv: Optional[List[str]] = None) -> int:
    """Argparse-driven entry point.

    Returns
    -------
    int
        ``0`` on success, ``2`` on bad input, ``3`` on a solver failure.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    materials = _resolve_via_shim('MATERIALS', MATERIALS)

    if args.list_materials:
        for name in sorted(materials):
            print(name)
        return 0

    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose are mutually exclusive.")

    _configure_cli_logging(quiet=args.quiet, verbose=args.verbose)

    if args.material not in materials:
        parser.error(
            f"Unknown material {args.material!r}. "
            f"Available presets: {sorted(materials)}."
        )

    for Vp in args.vp:
        if not (0.0 <= Vp <= 1.0) or Vp != Vp:  # NaN-safe range check
            parser.error(
                f"--vp value {Vp!r} is out of range; expected a finite "
                f"float in [0, 1] (a void *fraction*, not a percentage)."
            )

    output_dir = args.output_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create output directory {output_dir!r}: {exc}",
              file=sys.stderr)
        return 2

    # Back-compat shim: a few callers may still reference _log. Now that
    # progress is routed through ``logger``, this just forwards to
    # logger.info so --quiet (WARNING level) silences these too.
    def _log(msg: str) -> None:
        logger.info("%s", msg)

    # Resolve via the shim so tests that monkeypatch
    # ``porosity_fe_analysis.compare_configurations`` (and POROSITY_CONFIGS)
    # see their patch honoured at call time (#119).
    cmp_fn = _resolve_via_shim('compare_configurations', compare_configurations)
    save_fn = _resolve_via_shim('save_results_to_json', save_results_to_json)
    viz = _resolve_via_shim('FEVisualizer', FEVisualizer)
    porosity_configs = _resolve_via_shim('POROSITY_CONFIGS', POROSITY_CONFIGS)

    all_results = {}
    for Vp in args.vp:
        Vp_label = _vp_label(Vp)
        try:
            # ``return_artifacts=True`` because the --plots path needs the
            # live mesh / empirical_solver / porosity_field objects for
            # the FEVisualizer calls below (#44 item 3 migration).
            results, artifacts = cmp_fn(
                Vp,
                material_name=args.material,
                applied_stress=args.applied_stress,
                seed=args.seed,
                n_jobs=args.jobs,
                return_artifacts=True,
            )
        except ValueError as exc:
            print(f"ERROR: bad input for Vp={Vp}: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - surface as solver failure
            print(f"ERROR: solver failure for Vp={Vp}: {exc}", file=sys.stderr)
            return 3
        all_results[Vp_label] = results

        if args.plots:
            for name in results:
                art = artifacts[name]
                viz.plot_porosity_field(
                    art.porosity_field,
                    save_path=os.path.join(
                        output_dir, f"porosity_profile_{name}_{Vp_label}.png"))
                viz.plot_mesh_3d(
                    art.mesh,
                    save_path=os.path.join(
                        output_dir, f"porosity_mesh_3d_{name}_{Vp_label}.png"))
                viz.plot_mesh_detail(
                    art.mesh,
                    save_path=os.path.join(
                        output_dir, f"porosity_mesh_detail_{name}_{Vp_label}.png"))
                viz.plot_damage_contour(
                    art.mesh,
                    art.empirical_solver,
                    save_path=os.path.join(
                        output_dir, f"porosity_damage_{name}_{Vp_label}.png"))
            viz.plot_model_comparison(
                results,
                save_path=os.path.join(
                    output_dir, f"porosity_comparison_{Vp_label}.png"))

        out_path = os.path.join(
            output_dir, f"porosity_analysis_results_{Vp_label}.json")
        save_fn(results, out_path, artifacts=artifacts)

    if args.plots and all_results:
        viz.plot_knockdown_curves(
            all_results,
            save_path=os.path.join(output_dir, "porosity_knockdown_curves.png"))

    _bar = "=" * 70
    logger.info("\n%s", _bar)
    logger.info("COMPLETE ANALYSIS FINISHED")
    logger.info("%s", _bar)
    logger.info("Material: %s", args.material)
    logger.info("Porosity levels analyzed: %s",
                [f"{v*100:.2f}%" for v in args.vp])
    logger.info("Configurations: %s", list(porosity_configs.keys()))
    logger.info("Output directory: %s", os.path.abspath(output_dir))
    return 0
